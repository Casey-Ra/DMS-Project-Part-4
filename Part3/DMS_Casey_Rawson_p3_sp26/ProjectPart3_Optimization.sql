CREATE INDEX idx_customerhealthprofile_customer
ON "CustomerHealthProfile" ("CustomerID");

CREATE INDEX idx_customerhealthprofile_disease
ON "CustomerHealthProfile" ("DiseaseID");

CREATE INDEX idx_customerhealthprofile_riskfactor
ON "CustomerHealthProfile" ("PrimaryRiskFactorID");

CREATE INDEX idx_healthdatalakeref_profile
ON "HealthDataLakeRef" ("HealthProfileID");

CREATE INDEX idx_healthdatalakeref_source
ON "HealthDataLakeRef" ("SourceSystem");

CREATE INDEX idx_contract_customer_role_customer
ON "CustomerContractRole" ("CustomerID");

CREATE INDEX idx_contract_customer_role_contract
ON "CustomerContractRole" ("ContractID");

CREATE INDEX idx_contractbenefit_contract
ON "ContractBenefit" ("ContractID");

CREATE INDEX idx_contractpremium_benefit
ON "ContractPremium" ("ContractBenefitID");

CREATE MATERIALIZED VIEW "CustomerHealthRiskSummary" AS
SELECT
  c."CustomerID",
  c."CustomerType",
  c."FirstName",
  c."LastName",
  c."OrganizationName",
  hp."HealthProfileID",
  cd."DiseaseName",
  rf."FactorName" AS "PrimaryRiskFactor",
  hp."ChronicDiseaseRiskScore",
  hp."AssessmentDate",
  dl."SourceSystem",
  dl."DatasetName",
  dl."CloudStorageURI"
FROM "CustomerHealthProfile" hp
JOIN "Customer" c
  ON hp."CustomerID" = c."CustomerID"
JOIN "ChronicDisease" cd
  ON hp."DiseaseID" = cd."DiseaseID"
LEFT JOIN "RiskFactor" rf
  ON hp."PrimaryRiskFactorID" = rf."RiskFactorID"
LEFT JOIN "HealthDataLakeRef" dl
  ON hp."HealthProfileID" = dl."HealthProfileID";

CREATE INDEX idx_customerhealthrisksummary_customer
ON "CustomerHealthRiskSummary" ("CustomerID");