import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def load_data(filename):
    data = pd.read_csv(f'dataset/{filename}')
    return data.iloc[:, :-1], data.iloc[:, -1]

def train_model(X, y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)
    return model, le

def load_description_data():
    # Load all dataframes and standardize column names
    description_df = pd.read_csv('dataset/description.csv')
    precautions_df = pd.read_csv('dataset/precautions_df.csv')
    medications_df = pd.read_csv('dataset/medications.csv')
    diets_df = pd.read_csv('dataset/diets.csv')
    workout_df = pd.read_csv('dataset/workout_df.csv')
    
    # Standardize column names (some files use 'Disease', others use 'disease')
    if 'disease' in workout_df.columns:
        workout_df = workout_df.rename(columns={'disease': 'Disease'})
    if 'disease' in precautions_df.columns:
        precautions_df = precautions_df.rename(columns={'disease': 'Disease'})
    if 'disease' in medications_df.columns:
        medications_df = medications_df.rename(columns={'disease': 'Disease'})
    if 'disease' in diets_df.columns:
        diets_df = diets_df.rename(columns={'disease': 'Disease'})
    
    # Clean up any unnamed columns
    for df in [description_df, precautions_df, medications_df, diets_df, workout_df]:
        unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
        if unnamed_cols:
            df.drop(columns=unnamed_cols, inplace=True)
    
    return description_df, precautions_df, medications_df, diets_df, workout_df

def get_disease_severity(disease):
    """Get the severity level of a disease based on its typical presentation."""
    # Define severity levels for diseases (1 = mild, 2 = moderate, 3 = severe)
    severity_mapping = {
        # Mild conditions (common, generally not dangerous)
        'Allergy': 1,
        'Common Cold': 1,
        'Acne': 1,
        'Sinusitis': 1,
        'Migraine': 1,
        'Gastritis': 1,
        
        # Moderate conditions (need attention but not immediately life-threatening)
        'Bronchial Asthma': 2,
        'Hypertension': 2,
        'Cervical spondylosis': 2,
        'Hyperthyroidism': 2,
        'Hypothyroidism': 2,
        'Gastroenteritis': 2,
        'Arthritis': 2,
        
        # Severe conditions (require immediate medical attention)
        'Heart attack': 3,
        'Paralysis (brain hemorrhage)': 3,
        'Pneumonia': 3,
        'Tuberculosis': 3,
        'Malaria': 3,
        'Dengue': 3,
        'Typhoid': 3,
        'hepatitis A': 3,
        'Hepatitis B': 3,
        'Hepatitis C': 3,
        'Hepatitis D': 3,
        'Hepatitis E': 3,
        'Alcoholic hepatitis': 3,
        'Jaundice': 3,
    }
    
    return severity_mapping.get(disease, 2)  # Default to moderate if not found

def get_top_predictions(model, le, symptoms, n=5):
    """Get predictions sorted by probability and severity level.
    Returns more predictions from mild and moderate conditions."""
    pred_probs = model.predict_proba(symptoms)
    
    # Get all predictions with their probabilities
    predictions = []
    for idx, prob in enumerate(pred_probs[0]):
        disease = le.inverse_transform([idx])[0]
        severity = get_disease_severity(disease)
        predictions.append((disease, prob, severity))
    
    # Sort predictions:
    # 1. First by probability within each severity level
    # 2. Then by severity level (mild → moderate → severe)
    severity_groups = {1: [], 2: [], 3: []}
    
    for disease, prob, severity in predictions:
        severity_groups[severity].append((disease, prob))
    
    # Sort within each severity group by probability
    for severity in severity_groups:
        severity_groups[severity].sort(key=lambda x: x[1], reverse=True)
    
    # Combine results, taking top predictions from each severity level
    final_predictions = []
    
    # Show more mild conditions (top 3)
    if severity_groups[1]:
        final_predictions.extend(severity_groups[1][:3])
    
    # Show moderate conditions (top 2)
    if severity_groups[2]:
        final_predictions.extend(severity_groups[2][:2])
    
    # Only show severe conditions if probability is high (>= 0.3)
    if severity_groups[3]:
        severe_conditions = [(d, p) for d, p in severity_groups[3] if p >= 0.3]
        if severe_conditions:
            final_predictions.extend(severe_conditions[:1])
    
    # Sort final list by probability and take top N
    final_predictions.sort(key=lambda x: x[1], reverse=True)
    return final_predictions[:n]

def get_disease_info(disease, description_df, precautions_df, medications_df, diets_df, workout_df):
    info = {
        "description": "",
        "precautions": [],
        "medications": [],
        "diets": [],
        "workout": []
    }
    
    try:
        # Get description
        if not description_df.empty and 'Disease' in description_df.columns:
            matching_desc = description_df[description_df['Disease'] == disease]
            if not matching_desc.empty:
                info["description"] = matching_desc['Description'].iloc[0]
        
        # Get precautions
        if not precautions_df.empty:
            matching_prec = precautions_df[precautions_df['Disease'] == disease]
            if not matching_prec.empty:
                precautions = matching_prec.iloc[0]
                info["precautions"] = [p for p in precautions[1:] if pd.notna(p)]
        
        # Get medications
        if not medications_df.empty:
            matching_med = medications_df[medications_df['Disease'] == disease]
            if not matching_med.empty:
                medications = matching_med.iloc[0]
                info["medications"] = [m for m in medications[1:] if pd.notna(m)]
        
        # Get diets
        if not diets_df.empty:
            matching_diet = diets_df[diets_df['Disease'] == disease]
            if not matching_diet.empty:
                diets = matching_diet.iloc[0]
                info["diets"] = [d for d in diets[1:] if pd.notna(d)]
        
        # Get workout
        if not workout_df.empty:
            matching_workout = workout_df[workout_df['Disease'] == disease]
            if not matching_workout.empty:
                workouts = matching_workout['workout'].tolist()
                info["workout"] = [w for w in workouts if pd.notna(w)]
    
    except Exception as e:
        print(f"Warning: Error getting information for {disease}: {str(e)}")
    
    return info

def print_disease_info(disease, info):
    """Print disease information organized by severity levels."""
    severity = get_disease_severity(disease)
    severity_text = {
        1: "MILD LEVEL",
        2: "MODERATE LEVEL",
        3: "SEVERE LEVEL"
    }
    
    print(f"\n================== {severity_text[severity]} ==================")
    print(f"Condition: {disease}")
    
    if severity == 3:
        print("\nIMPORTANT: Please consult a healthcare provider immediately.")
    
    if info["precautions"]:
        print("\nPrecautions:")
        for i, precaution in enumerate(info["precautions"], 1):
            print(f"{i}. {precaution}")
    
    if info["medications"]:
        print("\nRecommended Medications:")
        for i, medication in enumerate(info["medications"], 1):
            print(f"{i}. {medication}")
    
    if info["diets"]:
        print("\nDietary Recommendations:")
        for i, diet in enumerate(info["diets"], 1):
            print(f"{i}. {diet}")
    
    if info["workout"]:
        print("\nLifestyle & Exercise:")
        for i, workout in enumerate(info["workout"], 1):
            print(f"{i}. {workout}")

def normalize_symptom(symptom):
    """Map common symptom variations to their standard names"""
    symptom_mapping = {
        # Temperature related
        'fever': 'mild_fever',
        'high temperature': 'high_fever',
        'temperature': 'mild_fever',
        
        # Pain related
        'stomach ache': 'stomach_pain',
        'belly ache': 'belly_pain',
        'throat pain': 'throat_irritation',
        
        # Common variations
        'coughing': 'cough',
        'tired': 'fatigue',
        'exhausted': 'fatigue',
        'throwing up': 'vomiting',
        'dizzy': 'dizziness',
        'feeling weak': 'weakness_in_limbs',
        'cant breathe': 'breathlessness',
        'trouble breathing': 'breathlessness',
        'short of breath': 'breathlessness',
        'sweating a lot': 'sweating',
        'excessive sweating': 'sweating',
        'feeling sick': 'nausea',
        'feeling nauseous': 'nausea',
        'runny nose': 'continuous_sneezing'
    }
    return symptom_mapping.get(symptom.lower(), symptom.lower())

def get_severity_description(probability):
    """Convert probability to severity description"""
    if probability >= 0.7:
        return "Very Likely"
    elif probability >= 0.5:
        return "Likely"
    elif probability >= 0.3:
        return "Possible"
    elif probability >= 0.1:
        return "Less Likely"
    else:
        return "Unlikely"

def get_common_recommendations(diseases, description_df, precautions_df, medications_df, diets_df, workout_df):
    """Get common recommendations that would help with any of the predicted conditions."""
    common_info = {
        "precautions": set(),
        "medications": set(),
        "diets": set(),
        "workout": set()
    }
    
    for disease in diseases:
        try:
            # Get precautions
            if not precautions_df.empty:
                matching_prec = precautions_df[precautions_df['Disease'] == disease]
                if not matching_prec.empty:
                    precautions = matching_prec.iloc[0]
                    common_info["precautions"].update([p for p in precautions[1:] if pd.notna(p)])
            
            # Get diets
            if not diets_df.empty:
                matching_diet = diets_df[diets_df['Disease'] == disease]
                if not matching_diet.empty:
                    diets = matching_diet.iloc[0]
                    common_info["diets"].update([d for d in diets[1:] if pd.notna(d)])
            
            # Get workout
            if not workout_df.empty:
                matching_workout = workout_df[workout_df['Disease'] == disease]
                if not matching_workout.empty:
                    workouts = matching_workout['workout'].tolist()
                    common_info["workout"].update([w for w in workouts if pd.notna(w)])
        
        except Exception as e:
            print(f"Warning: Error getting information for {disease}: {str(e)}")
    
    # Convert sets back to lists
    return {k: list(v) for k, v in common_info.items()}

def print_recommendations(severity_groups):
    """Print conditions by severity level and general recommendations."""
    print("\n=================== POSSIBLE CONDITIONS ===================")
    
    # Print conditions by severity
    for severity_level in ["MILD LEVEL", "MODERATE LEVEL", "SEVERE LEVEL"]:
        if severity_groups[severity_level]:
            print(f"\n{severity_level}:")
            for disease, likelihood in severity_groups[severity_level]:
                print(f"- {disease} ({likelihood})")
    
    if severity_groups["SEVERE LEVEL"]:
        print("\nCAUTION: If symptoms persist or worsen, please consult a healthcare provider immediately.")

def main():
    # Load and train the model
    print("Loading and training the model...")
    X, y = load_data('Training-weighted.csv')
    model, le = train_model(X, y)
    
    # Load all the supplementary data
    print("Loading disease information...")
    description_df, precautions_df, medications_df, diets_df, workout_df = load_description_data()
    
    print("\nExample symptoms you can enter:")
    print("- fever, headache, fatigue")
    print("- cough, fever, breathlessness")
    print("- stomach pain, nausea, vomiting")
    
    while True:
        print("\nEnter your symptoms (comma-separated), or 'quit' to exit:")
        user_input = input().strip()
        
        if user_input.lower() == 'quit':
            break
        
        # Create symptom vector
        symptoms = [normalize_symptom(s.strip()) for s in user_input.split(',')]
        symptom_vector = pd.DataFrame(0, index=[0], columns=X.columns)
        
        valid_symptoms = []
        for symptom in symptoms:
            if symptom in symptom_vector.columns:
                symptom_vector[symptom] = 1
                valid_symptoms.append(symptom)
            else:
                print(f"Note: '{symptom}' was not recognized - please check spelling or try a different term")
        
        if not valid_symptoms:
            print("No valid symptoms entered. Please try again with different symptoms.")
            continue
        
        print("\nAnalyzing symptoms:", ", ".join(valid_symptoms))
        
        # Get prediction
        top_predictions = get_top_predictions(model, le, symptom_vector, n=5)
        
        # Group by severity
        severity_groups = {"MILD LEVEL": [], "MODERATE LEVEL": [], "SEVERE LEVEL": []}
        all_predicted_diseases = []
        
        for disease, prob in top_predictions:
            severity = get_disease_severity(disease)
            likelihood = get_severity_description(prob)
            all_predicted_diseases.append(disease)
            
            if severity == 1:
                severity_groups["MILD LEVEL"].append((disease, likelihood))
            elif severity == 2:
                severity_groups["MODERATE LEVEL"].append((disease, likelihood))
            else:
                severity_groups["SEVERE LEVEL"].append((disease, likelihood))
        
        # Print conditions by severity
        print_recommendations(severity_groups)
        
        # Get and print common recommendations
        common_recommendations = get_common_recommendations(all_predicted_diseases, description_df, precautions_df, medications_df, diets_df, workout_df)
        
        print("\n=============== GENERAL RECOMMENDATIONS ===============")
        print("These recommendations may help relieve your symptoms:")
        
        if common_recommendations["precautions"]:
            print("\nPrecautions:")
            for i, precaution in enumerate(common_recommendations["precautions"], 1):
                print(f"{i}. {precaution}")
        
        if common_recommendations["diets"]:
            print("\nDietary Recommendations:")
            for i, diet in enumerate(common_recommendations["diets"], 1):
                print(f"{i}. {diet}")
        
        if common_recommendations["workout"]:
            print("\nLifestyle & Exercise:")
            for i, workout in enumerate(common_recommendations["workout"], 1):
                print(f"{i}. {workout}")
        
        print("\nNOTE: These are general recommendations only.")
        print("For specific treatment, please consult a healthcare provider.")

if __name__ == "__main__":
    main() 