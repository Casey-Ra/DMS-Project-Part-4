SELECT
  "CustomerID" AS "ID",
  COALESCE("FirstName" || ' ' || "LastName", "OrganizationName") AS "Customer",
  "DiseaseName" AS "Disease",
  "PrimaryRiskFactor" AS "Risk Factor",
  "ChronicDiseaseRiskScore" AS "Risk Score",
  "SourceSystem" AS "Source",
  "DatasetName" AS "Dataset"
FROM "CustomerHealthRiskSummary"
ORDER BY "AssessmentDate" DESC;