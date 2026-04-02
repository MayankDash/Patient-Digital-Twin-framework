FEATURES = [
    "Age",
    "Gender",
    "BMI",
    "Systolic_BP",
    "Diastolic_BP",
    "Glucose",
    "Insulin",
    "Total_Cholesterol",
    "HDL_Cholesterol",
]

TARGET = "Metabolic_Risk"
RAW_TARGET = "HbA1c"

# NHANES coding in DEMO_J.xpt: 1=Male, 2=Female. Notebook maps: 1->0, 2->1
GENDER_MAPPING = {1.0: 0, 2.0: 1, 1: 0, 2: 1}
