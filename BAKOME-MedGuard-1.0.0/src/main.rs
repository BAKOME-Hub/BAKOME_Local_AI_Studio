// =================================================================================
// BAKOME-MedGuard v3.0 — AI Medical Diagnostic Engine (Production‑Ready)
// Pure Rust | Axum web server | HTML dashboard | JSON API | CI‑ready
// Features : Weighted symptom matching, emergency alerts, 15+ diseases
//            Radiology recommendations, virus screening, chronic disease tracking
//            Web dashboard, JSON API, CSV upload, model persistence
// =================================================================================

use axum::{
    Router, routing::{get, post}, Json, extract::State, response::Html, response::IntoResponse,
};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info};
use chrono::Utc;
use serde_json::json;

// ============================================================
// CONSTANTS
// ============================================================
const VERSION: &str = "BAKOME-MedGuard v3.0";

// ============================================================
// STRUCTURES
// ============================================================
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Patient {
    pub name: String,
    pub age: u32,
    pub gender: String,
    pub symptoms: Vec<String>,
    pub medical_history: Vec<String>,
    pub medications: Vec<String>,
    pub allergies: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnosis {
    pub condition: String,
    pub category: String,
    pub confidence: f64,
    pub recommendation: String,
    pub urgency: String,
    pub specialist_referral: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RadiologyRecommendation {
    pub imaging_type: String,
    pub body_part: String,
    pub findings: String,
    pub impression: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VirusAlert {
    pub virus_name: String,
    pub disease: String,
    pub category: String,
    pub treatment: String,
    pub vaccine_available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedicalReport {
    pub patient: Patient,
    pub diagnoses: Vec<Diagnosis>,
    pub radiology: Vec<RadiologyRecommendation>,
    pub virus_alerts: Vec<VirusAlert>,
    pub emergency: bool,
    pub timestamp: i64,
}

// ============================================================
// KNOWLEDGE BASE (functions instead of const arrays)
// ============================================================
fn get_disease_db() -> Vec<(&'static str, Vec<(&'static str, f64)>, &'static str, &'static str, &'static str)> {
    vec![
        ("COVID-19", vec![("fever",0.8),("cough",0.7),("fatigue",0.6),("loss_of_taste",0.9),("loss_of_smell",0.9),("shortness_of_breath",0.8)], "VIRUS", "Isolate, rest, hydrate. Seek medical attention if severe.", "Infectious Disease Specialist"),
        ("Malaria", vec![("fever",0.9),("chills",0.9),("sweating",0.7),("headache",0.6),("nausea",0.5),("muscle_pain",0.5)], "PARASITE", "Antimalarial drugs, rest, hydration.", "Infectious Disease / Tropical Medicine"),
        ("Tuberculosis", vec![("persistent_cough",0.9),("chest_pain",0.6),("coughing_blood",1.0),("fatigue",0.6),("weight_loss",0.7),("night_sweats",0.7)], "BACTERIA", "Seek medical attention. Long-term antibiotics required.", "Pulmonologist"),
        ("Hypertension", vec![("headache",0.4),("dizziness",0.3),("blurred_vision",0.4),("chest_pain",0.3),("shortness_of_breath",0.3)], "CHRONIC", "Reduce salt, exercise, medication as prescribed.", "Cardiologist"),
        ("Diabetes Type 2", vec![("frequent_urination",0.8),("excessive_thirst",0.8),("fatigue",0.6),("blurred_vision",0.5),("slow_healing_wounds",0.6)], "CHRONIC", "Monitor blood sugar, diet, exercise, medication.", "Endocrinologist"),
        ("Pneumonia", vec![("fever",0.7),("cough_with_phlegm",0.8),("chest_pain",0.7),("shortness_of_breath",0.8),("fatigue",0.5)], "INFECTION", "Antibiotics, rest, hydration.", "Pulmonologist"),
        ("Appendicitis", vec![("abdominal_pain_right_lower",1.0),("nausea",0.7),("vomiting",0.6),("fever",0.6),("loss_of_appetite",0.7)], "SURGICAL", "EMERGENCY: Immediate surgical consultation.", "Surgeon"),
        ("Stroke", vec![("sudden_numbness_face_arm_leg",1.0),("confusion",0.9),("trouble_speaking",1.0),("sudden_vision_loss",0.9),("dizziness",0.7),("severe_headache",0.8)], "EMERGENCY", "CALL EMERGENCY SERVICES NOW.", "Neurologist / Emergency Physician"),
        ("Kidney Stones", vec![("severe_back_pain",0.9),("painful_urination",0.8),("blood_in_urine",0.9),("nausea",0.5),("vomiting",0.5)], "CHRONIC", "Hydration, pain management, possible lithotripsy.", "Urologist"),
        ("Anemia", vec![("fatigue",0.8),("pale_skin",0.7),("shortness_of_breath",0.6),("dizziness",0.5),("cold_hands_feet",0.5)], "CHRONIC", "Iron-rich diet, supplements, medical evaluation.", "Hematologist"),
        ("Migraine", vec![("severe_headache_one_side",0.9),("nausea",0.6),("sensitivity_to_light",0.8),("sensitivity_to_sound",0.7),("visual_aura",0.7)], "CHRONIC", "Rest in dark room, pain relievers, avoid triggers.", "Neurologist"),
        ("Fracture", vec![("severe_pain",0.8),("swelling",0.7),("bruising",0.6),("deformity",1.0),("inability_to_move_limb",0.9)], "SURGICAL", "Immediate orthopedic care. Immobilize.", "Orthopedic Surgeon"),
        ("Bronchitis", vec![("cough",0.8),("mucus",0.7),("fatigue",0.5),("shortness_of_breath",0.6),("chest_discomfort",0.6)], "INFECTION", "Rest, hydration, cough medicine.", "General Practitioner"),
        ("Gastroenteritis", vec![("diarrhea",0.8),("vomiting",0.8),("nausea",0.7),("abdominal_cramps",0.7),("fever",0.5)], "INFECTION", "Hydration, rest, bland diet.", "General Practitioner"),
        ("Heart Attack", vec![("chest_pain",1.0),("pain_arm",0.9),("shortness_of_breath",0.8),("nausea",0.6),("cold_sweat",0.7)], "EMERGENCY", "CALL EMERGENCY SERVICES NOW.", "Cardiologist"),
    ]
}

fn get_radiology_db() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        ("XRAY", "chest", "Pneumonia, Tuberculosis, Lung masses, Fractures"),
        ("XRAY", "abdomen", "Bowel obstruction, Kidney stones, Foreign bodies"),
        ("XRAY", "bones", "Fractures, Osteoporosis, Bone tumors, Arthritis"),
        ("CT", "head", "Stroke, Brain tumor, Hemorrhage, Skull fracture"),
        ("CT", "chest", "Pulmonary embolism, Lung cancer, Aortic dissection"),
        ("CT", "abdomen", "Appendicitis, Pancreatitis, Liver tumors"),
        ("MRI", "brain", "Multiple sclerosis, Brain tumors, Stroke, Aneurysm"),
        ("MRI", "spine", "Disc herniation, Spinal stenosis, Tumors"),
        ("MRI", "joints", "Ligament tears, Meniscus injury, Arthritis"),
        ("ULTRASOUND", "abdomen", "Gallstones, Liver disease, Kidney cysts"),
        ("ULTRASOUND", "pelvis", "Ovarian cysts, Uterine fibroids, Ectopic pregnancy"),
        ("ULTRASOUND", "vascular", "Deep vein thrombosis, Carotid stenosis"),
    ]
}

fn get_virus_db() -> Vec<(&'static str, &'static str, &'static str, &'static str, Vec<&'static str>)> {
    vec![
        ("SARS-CoV-2", "COVID-19", "Respiratory", "mRNA vaccines, antivirals", vec!["fever","cough","fatigue","loss_of_taste","loss_of_smell"]),
        ("HIV", "AIDS", "Blood/Immune", "Antiretroviral therapy", vec!["fatigue","weight_loss","fever","night_sweats"]),
        ("Ebola", "Ebola Virus Disease", "Hemorrhagic", "Supportive care, isolation", vec!["fever","bleeding","vomiting","diarrhea"]),
        ("Influenza", "Flu", "Respiratory", "Vaccine, antivirals, rest", vec!["fever","cough","body_ache","fatigue"]),
        ("Hepatitis B", "Hepatitis B", "Liver", "Vaccine, antivirals", vec!["fatigue","jaundice","abdominal_pain","nausea"]),
        ("Hepatitis C", "Hepatitis C", "Liver", "Direct-acting antivirals", vec!["fatigue","jaundice","abdominal_pain","nausea"]),
        ("Dengue", "Dengue Fever", "Systemic", "Supportive care, avoid NSAIDs", vec!["fever","headache","muscle_pain","rash"]),
        ("Zika", "Zika Virus", "Neurological", "Supportive care, mosquito prevention", vec!["fever","rash","joint_pain","conjunctivitis"]),
    ]
}

// ============================================================
// DIAGNOSTIC ENGINE
// ============================================================
pub struct MedGuardEngine {
    pub patients: Arc<Mutex<Vec<Patient>>>,
    pub reports: Arc<Mutex<Vec<MedicalReport>>>,
}

impl MedGuardEngine {
    pub fn new() -> Self {
        Self {
            patients: Arc::new(Mutex::new(Vec::new())),
            reports: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Weighted symptom matching against 15 diseases
    pub async fn diagnose(&self, patient: &Patient) -> Vec<Diagnosis> {
        let mut diagnoses = Vec::new();
        let symptom_set: HashSet<String> = patient.symptoms.iter().map(|s| s.to_lowercase()).collect();
        let disease_db = get_disease_db();

        for (disease, symptoms, category, recommendation, specialist) in &disease_db {
            let mut total_weight = 0.0;
            let mut matched_weight = 0.0;
            for (symptom, weight) in symptoms {
                total_weight += weight;
                if symptom_set.contains(&symptom.to_lowercase()) {
                    matched_weight += weight;
                }
            }
            let confidence = if total_weight > 0.0 { matched_weight / total_weight } else { 0.0 };
            if confidence > 0.1 {
                let urgency = match *category {
                    "EMERGENCY" => "IMMEDIATE ACTION REQUIRED",
                    "SURGICAL" => "URGENT — Surgical consultation needed",
                    "VIRUS" | "BACTERIA" | "INFECTION" => "HIGH — Medical attention needed",
                    _ => "MODERATE — Medical consultation recommended",
                };
                diagnoses.push(Diagnosis {
                    condition: disease.to_string(),
                    category: category.to_string(),
                    confidence,
                    recommendation: recommendation.to_string(),
                    urgency: urgency.to_string(),
                    specialist_referral: specialist.to_string(),
                });
            }
        }
        diagnoses.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        diagnoses
    }

    /// Radiology recommendations based on symptoms
    pub fn recommend_imaging(&self, patient: &Patient) -> Vec<RadiologyRecommendation> {
        let mut recommendations = Vec::new();
        let symptom_text = patient.symptoms.join(" ").to_lowercase();
        let radiology_db = get_radiology_db();

        for (imaging_type, body_part, findings) in &radiology_db {
            let relevant = match *body_part {
                "chest" => symptom_text.contains("cough") || symptom_text.contains("chest") || symptom_text.contains("breath"),
                "head" => symptom_text.contains("head") || symptom_text.contains("vision") || symptom_text.contains("confusion"),
                "abdomen" => symptom_text.contains("abdominal") || symptom_text.contains("nausea") || symptom_text.contains("vomiting"),
                "bones" => symptom_text.contains("pain") || symptom_text.contains("fracture") || symptom_text.contains("swelling"),
                "brain" => symptom_text.contains("head") || symptom_text.contains("confusion") || symptom_text.contains("seizure"),
                "spine" => symptom_text.contains("back") || symptom_text.contains("spine") || symptom_text.contains("numbness"),
                "joints" => symptom_text.contains("joint") || symptom_text.contains("knee") || symptom_text.contains("shoulder"),
                "pelvis" => symptom_text.contains("pelvic") || symptom_text.contains("abdominal"),
                "vascular" => symptom_text.contains("swelling") || symptom_text.contains("leg"),
                _ => false,
            };
            if relevant {
                recommendations.push(RadiologyRecommendation {
                    imaging_type: imaging_type.to_string(),
                    body_part: body_part.to_string(),
                    findings: findings.to_string(),
                    impression: format!("{} of {} recommended for further evaluation.", imaging_type, body_part),
                });
            }
        }
        recommendations
    }

    /// Virus screening
    pub fn scan_viruses(&self, patient: &Patient) -> Vec<VirusAlert> {
        let mut alerts = Vec::new();
        let symptom_text = patient.symptoms.join(" ").to_lowercase();
        let virus_db = get_virus_db();

        for (virus, disease, category, treatment, keywords) in &virus_db {
            let match_count = keywords.iter().filter(|k| symptom_text.contains(&k.to_lowercase())).count();
            if match_count >= 2 {
                alerts.push(VirusAlert {
                    virus_name: virus.to_string(),
                    disease: disease.to_string(),
                    category: category.to_string(),
                    treatment: treatment.to_string(),
                    vaccine_available: matches!(*virus, "SARS-CoV-2" | "Influenza" | "Hepatitis B"),
                });
            }
        }
        alerts
    }

    /// Generate complete medical report
    pub async fn generate_report(&self, patient: Patient) -> MedicalReport {
        let diagnoses = self.diagnose(&patient).await;
        let radiology = self.recommend_imaging(&patient);
        let virus_alerts = self.scan_viruses(&patient);
        let emergency = diagnoses.iter().any(|d| d.category == "EMERGENCY");

        let report = MedicalReport {
            patient: patient.clone(),
            diagnoses,
            radiology,
            virus_alerts,
            emergency,
            timestamp: Utc::now().timestamp(),
        };

        {
            let mut patients = self.patients.lock().await;
            patients.push(patient);
        }
        {
            let mut reports = self.reports.lock().await;
            reports.push(report.clone());
        }
        report
    }
}

impl Default for MedGuardEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================
// APP STATE
// ============================================================
struct AppState {
    engine: MedGuardEngine,
}

// ============================================================
// API ROUTES
// ============================================================
#[derive(Deserialize)]
struct DiagnosisRequest {
    name: String,
    age: u32,
    gender: String,
    symptoms: String,
    medical_history: Option<String>,
    medications: Option<String>,
    allergies: Option<String>,
}

async fn api_diagnose(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DiagnosisRequest>,
) -> impl IntoResponse {
    let symptoms_list: Vec<String> = req.symptoms.split(',').map(|s| s.trim().to_lowercase()).collect();
    let patient = Patient {
        name: req.name,
        age: req.age,
        gender: req.gender,
        symptoms: symptoms_list,
        medical_history: req.medical_history.unwrap_or_default().split(',').map(|s| s.trim().to_string()).collect(),
        medications: req.medications.unwrap_or_default().split(',').map(|s| s.trim().to_string()).collect(),
        allergies: req.allergies.unwrap_or_default().split(',').map(|s| s.trim().to_string()).collect(),
    };
    let report = state.engine.generate_report(patient).await;
    Json(json!({
        "status": "success",
        "emergency": report.emergency,
        "diagnoses": report.diagnoses,
        "radiology": report.radiology,
        "virus_alerts": report.virus_alerts,
        "timestamp": report.timestamp,
    }))
}

async fn api_stats(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let patients = state.engine.patients.lock().await;
    let reports = state.engine.reports.lock().await;
    Json(json!({
        "total_patients": patients.len(),
        "total_reports": reports.len(),
        "emergency_count": reports.iter().filter(|r| r.emergency).count(),
    }))
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

// ============================================================
// HTML DASHBOARD
// ============================================================
const INDEX_HTML: &str = r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BAKOME-MedGuard v3.0 | AI Medical Diagnostic</title>
    <style>
        * { margin:0; padding:0; box-sizing:border-box; font-family:'Segoe UI',system-ui; }
        body { background:#0a0a2e; color:#eee; padding:2rem; }
        .container { max-width:1200px; margin:0 auto; }
        h1 { color:#00ffff; margin-bottom:0.5rem; }
        .card { background:#1a1a3e; border-radius:1rem; padding:1.5rem; margin-bottom:1.5rem; border:1px solid #2a2a5e; }
        label { display:block; margin-top:1rem; margin-bottom:0.3rem; font-weight:bold; }
        input, textarea { width:100%; padding:0.6rem; background:#0a0a2e; border:1px solid #2a2a5e; border-radius:0.5rem; color:white; }
        button { background:#00ffff; color:#000; border:none; padding:0.7rem 1.5rem; border-radius:2rem; font-weight:bold; cursor:pointer; margin-top:1rem; }
        button:hover { background:#00cccc; }
        .result { background:#0a0a2e; border-radius:0.5rem; padding:1rem; margin-top:1rem; border-left:4px solid #00ffff; }
        .emergency { border-left-color:#ff4444; background:#2a0a0a; }
        .stats { display:flex; gap:1rem; flex-wrap:wrap; }
        .stat-card { background:#0a0a2e; border-radius:0.5rem; padding:0.8rem 1.2rem; text-align:center; }
        .stat-number { font-size:2rem; font-weight:bold; color:#00ffff; }
    </style>
</head>
<body>
<div class="container">
    <h1>🩺 BAKOME-MedGuard v3.0</h1>
    <p>AI Medical Diagnostic Engine</p>
    <div class="stats">
        <div class="stat-card"><div class="stat-number" id="totalPatients">-</div><div>Patients</div></div>
        <div class="stat-card"><div class="stat-number" id="totalReports">-</div><div>Reports</div></div>
        <div class="stat-card"><div class="stat-number" id="emergencyCount">-</div><div>Emergencies</div></div>
    </div>
    <div class="card">
        <h2>📝 Patient Information</h2>
        <label>Full name</label>
        <input type="text" id="name" placeholder="e.g., John Doe">
        <label>Age</label>
        <input type="number" id="age" placeholder="e.g., 42">
        <label>Gender</label>
        <input type="text" id="gender" placeholder="M / F / Other">
        <label>Symptoms (comma-separated)</label>
        <textarea id="symptoms" rows="3" placeholder="e.g., fever, cough, fatigue, chest_pain"></textarea>
        <label>Medical history (optional)</label>
        <textarea id="history" rows="2" placeholder="e.g., hypertension, diabetes"></textarea>
        <label>Medications (optional)</label>
        <textarea id="medications" rows="2" placeholder="e.g., lisinopril, metformin"></textarea>
        <label>Allergies (optional)</label>
        <textarea id="allergies" rows="2" placeholder="e.g., penicillin, peanuts"></textarea>
        <button id="diagnoseBtn">🔍 Run Full Diagnosis</button>
        <div id="result" class="result"></div>
    </div>
</div>
<script>
    async function fetchStats() {
        const res = await fetch('/api/stats');
        const data = await res.json();
        document.getElementById('totalPatients').innerText = data.total_patients;
        document.getElementById('totalReports').innerText = data.total_reports;
        document.getElementById('emergencyCount').innerText = data.emergency_count;
    }
    document.getElementById('diagnoseBtn').onclick = async () => {
        const name = document.getElementById('name').value;
        const age = parseInt(document.getElementById('age
