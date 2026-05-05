from __future__ import annotations

from datetime import date

from sqlalchemy import (
    Boolean,
    Date,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    create_engine,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "Customer"

    CustomerID: Mapped[int] = mapped_column(Integer, primary_key=True)
    CustomerType: Mapped[str] = mapped_column(String(30), nullable=False)
    FirstName: Mapped[str | None] = mapped_column(String(80))
    LastName: Mapped[str | None] = mapped_column(String(80))
    OrganizationName: Mapped[str | None] = mapped_column(String(150))
    DOB: Mapped[date | None] = mapped_column(Date)
    Status: Mapped[str | None] = mapped_column(String(30))


class ChronicDisease(Base):
    __tablename__ = "ChronicDisease"

    DiseaseID: Mapped[int] = mapped_column(Integer, primary_key=True)
    DiseaseName: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    DiseaseCategory: Mapped[str | None] = mapped_column(String(50))
    Description: Mapped[str | None] = mapped_column(String(255))


class RiskFactor(Base):
    __tablename__ = "RiskFactor"

    RiskFactorID: Mapped[int] = mapped_column(Integer, primary_key=True)
    FactorName: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    FactorCategory: Mapped[str | None] = mapped_column(String(50))
    IsChangeable: Mapped[bool | None] = mapped_column(Boolean)
    Description: Mapped[str | None] = mapped_column(String(255))


class CustomerHealthProfile(Base):
    __tablename__ = "CustomerHealthProfile"

    HealthProfileID: Mapped[int] = mapped_column(Integer, primary_key=True)
    CustomerID: Mapped[int] = mapped_column(ForeignKey('Customer.CustomerID'), nullable=False)
    DiseaseID: Mapped[int] = mapped_column(ForeignKey('ChronicDisease.DiseaseID'), nullable=False)
    ChronicDiseaseRiskScore: Mapped[float | None] = mapped_column(Numeric(5, 2))
    PrimaryRiskFactorID: Mapped[int | None] = mapped_column(ForeignKey('RiskFactor.RiskFactorID'))
    AssessmentDate: Mapped[date | None] = mapped_column(Date)
    AssessmentMethod: Mapped[str | None] = mapped_column(String(100))
    Notes: Mapped[str | None] = mapped_column(String(255))


class HealthDataLakeRef(Base):
    __tablename__ = "HealthDataLakeRef"

    DataLakeRefID: Mapped[int] = mapped_column(Integer, primary_key=True)
    HealthProfileID: Mapped[int] = mapped_column(
        ForeignKey('CustomerHealthProfile.HealthProfileID'),
        nullable=False,
    )
    SourceSystem: Mapped[str | None] = mapped_column(String(100))
    DataFormat: Mapped[str | None] = mapped_column(String(20))
    CloudStorageURI: Mapped[str] = mapped_column(String(500), nullable=False)
    DatasetName: Mapped[str | None] = mapped_column(String(150))
    LoadDate: Mapped[date | None] = mapped_column(Date)
    FileDescription: Mapped[str | None] = mapped_column(String(255))


class QuoteRecommendation(Base):
    __tablename__ = "QuoteRecommendation"

    QuoteRecommendationID: Mapped[int] = mapped_column(Integer, primary_key=True)
    CustomerID: Mapped[int] = mapped_column(ForeignKey('Customer.CustomerID'), nullable=False)
    HealthProfileID: Mapped[int] = mapped_column(
        ForeignKey('CustomerHealthProfile.HealthProfileID'),
        nullable=False,
    )
    RiskTier: Mapped[str] = mapped_column(String(20), nullable=False)
    BaseMonthlyRate: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    RateAdjustmentFactor: Mapped[float] = mapped_column(Numeric(5, 2), nullable=False)
    RecommendedMonthlyRate: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    RecommendationReason: Mapped[str | None] = mapped_column(Text)
    CreatedDate: Mapped[date] = mapped_column(Date, nullable=False)


def make_session(database_url: str) -> Session:
    engine = create_engine(database_url, future=True)
    Base.metadata.create_all(engine, tables=[QuoteRecommendation.__table__], checkfirst=True)
    return Session(engine)


def upsert_disease(session: Session) -> int:
    statement = (
        pg_insert(ChronicDisease)
        .values(
            DiseaseName="Lifestyle Chronic Disease Risk",
            DiseaseCategory="Lifestyle Risk",
            Description="Risk tier used for insurance rate recommendation and underwriting review.",
        )
        .on_conflict_do_update(
            index_elements=[ChronicDisease.DiseaseName],
            set_={
                "DiseaseCategory": "Lifestyle Risk",
                "Description": "Risk tier used for insurance rate recommendation and underwriting review.",
            },
        )
        .returning(ChronicDisease.DiseaseID)
    )
    return int(session.execute(statement).scalar_one())


def upsert_risk_factor(session: Session, factor_name: str) -> int:
    statement = (
        pg_insert(RiskFactor)
        .values(
            FactorName=factor_name,
            FactorCategory="Lifestyle",
            IsChangeable=factor_name not in ("Age",),
            Description="Primary lifestyle factor selected by the rate recommendation workflow.",
        )
        .on_conflict_do_update(
            index_elements=[RiskFactor.FactorName],
            set_={
                "FactorCategory": "Lifestyle",
                "IsChangeable": factor_name not in ("Age",),
                "Description": "Primary lifestyle factor selected by the rate recommendation workflow.",
            },
        )
        .returning(RiskFactor.RiskFactorID)
    )
    return int(session.execute(statement).scalar_one())


def refresh_health_summary(session: Session) -> None:
    session.execute(text('REFRESH MATERIALIZED VIEW "CustomerHealthRiskSummary";'))


def count_customer_health_profiles(session: Session, customer_id: int) -> int:
    statement = select(CustomerHealthProfile).where(
        CustomerHealthProfile.CustomerID == customer_id
    )
    return len(session.scalars(statement).all())
