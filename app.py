#!/usr/bin/env python3
"""
Clinical Assistant for Rare Disease Detection
Healthcare AI Hackathon Project

This Flask application provides an AI-powered clinical assistant
for detecting rare diseases like Aplastic Anemia through pattern
analysis of longitudinal patient data.
"""

from flask import Flask, request, jsonify, render_template, send_file
from datetime import datetime, timedelta
import re
import json
import sqlite3
from typing import Dict, List, Tuple, Any
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
import random
from scipy import stats

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'healthcare-ai-hackathon-2024'

class BoneMarrowAnalyzer:
    """Advanced bone marrow failure prediction system"""
    
    def __init__(self):
        self.hsc_healthy_threshold = 1.0        # ≥ 1% is healthy
        self.hsc_borderline_threshold = 0.5     # 0.5-1% is borderline
        self.hsc_concern_threshold = 0.5        # < 0.5% is high concern
        
        self.telomere_healthy_threshold = 10    # ≥ 10th percentile is healthy
        self.telomere_borderline_threshold = 1  # 1st-10th percentile is borderline  
        self.telomere_concern_threshold = 1     # < 1st percentile is high concern
        
        self.prediction_horizon = 6  # months
    
    def generate_mock_data(self, patient_type: str, months_back: int = 12) -> Dict[str, Any]:
        """Generate realistic mock data for different patient types"""
        dates = []
        hsc_values = []
        telomere_values = []
        
        # Generate dates every 2 months
        current_date = datetime.now()
        for i in range(0, months_back + 1, 2):
            dates.append(current_date - timedelta(days=i*30))
        
        dates.reverse()  # Oldest first
        
        if patient_type == "healthy":
            # Healthy patient - stable values above thresholds
            base_hsc = random.uniform(1.2, 2.0)
            base_telomere = random.uniform(25, 50)
            
            for i in range(len(dates)):
                noise_hsc = random.uniform(-0.1, 0.1)
                noise_telomere = random.uniform(-3, 3)
                hsc_values.append(max(1.1, base_hsc + noise_hsc))
                telomere_values.append(max(15, base_telomere + noise_telomere))
                
        elif patient_type == "borderline_declining":
            # Borderline patient with declining trend
            start_hsc = random.uniform(0.8, 0.95)
            start_telomere = random.uniform(8, 12)
            
            for i in range(len(dates)):
                # Linear decline with some noise
                decline_hsc = i * random.uniform(-0.03, -0.02)
                decline_telomere = i * random.uniform(-0.8, -0.5)
                noise_hsc = random.uniform(-0.02, 0.02)
                noise_telomere = random.uniform(-1, 1)
                
                hsc_values.append(max(0.3, start_hsc + decline_hsc + noise_hsc))
                telomere_values.append(max(0.5, start_telomere + decline_telomere + noise_telomere))
                
        elif patient_type == "high_concern":
            # High concern patient - already below thresholds
            base_hsc = random.uniform(0.2, 0.4)
            base_telomere = random.uniform(0.5, 2.0)
            
            for i in range(len(dates)):
                noise_hsc = random.uniform(-0.05, 0.05)
                noise_telomere = random.uniform(-0.3, 0.3)
                hsc_values.append(max(0.1, base_hsc + noise_hsc))
                telomere_values.append(max(0.1, base_telomere + noise_telomere))
        
        return {
            'dates': dates,
            'hsc_values': hsc_values,
            'telomere_values': telomere_values,
            'patient_type': patient_type
        }
    
    def calculate_trend_and_prediction(self, dates: List[datetime], values: List[float]) -> Dict[str, float]:
        """Calculate linear regression and predict time to threshold"""
        # Convert dates to months from first reading
        time_months = [(d - dates[0]).days / 30.44 for d in dates]
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_months, values)
        
        # Current time in months
        current_time = time_months[-1]
        current_value = values[-1]
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'current_time': current_time,
            'current_value': current_value,
            'trend_line': [slope * t + intercept for t in time_months]
        }
    
    def assess_patient_risk(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main risk assessment logic"""
        dates = patient_data['dates']
        hsc_values = patient_data['hsc_values']
        telomere_values = patient_data['telomere_values']
        
        # Calculate trends
        hsc_analysis = self.calculate_trend_and_prediction(dates, hsc_values)
        telomere_analysis = self.calculate_trend_and_prediction(dates, telomere_values)
        
        # Current values
        current_hsc = hsc_values[-1]
        current_telomere = telomere_values[-1]
        
        # Determine status
        hsc_status = self._get_hsc_status(current_hsc)
        telomere_status = self._get_telomere_status(current_telomere)
        
        # Check if second page should be triggered
        trigger_second_page = False
        trigger_reasons = []
        time_to_hsc_threshold = None
        time_to_telomere_threshold = None
        
        # HSC Assessment
        if hsc_status == "High Concern":
            trigger_second_page = True
            trigger_reasons.append("HSC below 0.5% (immediate concern)")
        elif hsc_status == "Borderline" and hsc_analysis['slope'] < 0:
            # Calculate time to reach 0.5% threshold
            time_to_threshold = self._calculate_time_to_threshold(
                hsc_analysis['slope'], 
                hsc_analysis['intercept'], 
                hsc_analysis['current_time'],
                self.hsc_concern_threshold
            )
            time_to_hsc_threshold = time_to_threshold
            if time_to_threshold <= self.prediction_horizon:
                trigger_second_page = True
                trigger_reasons.append(f"HSC projected to drop below 0.5% in {time_to_threshold:.1f} months")
        
        # Telomere Assessment  
        if telomere_status == "High Concern":
            trigger_second_page = True
            trigger_reasons.append("Telomere length below 1st percentile (immediate concern)")
        elif telomere_status == "Borderline" and telomere_analysis['slope'] < 0:
            # Calculate time to reach 1st percentile threshold
            time_to_threshold = self._calculate_time_to_threshold(
                telomere_analysis['slope'],
                telomere_analysis['intercept'],
                telomere_analysis['current_time'], 
                self.telomere_concern_threshold
            )
            time_to_telomere_threshold = time_to_threshold
            if time_to_threshold <= self.prediction_horizon:
                trigger_second_page = True
                trigger_reasons.append(f"Telomere length projected to drop below 1st percentile in {time_to_threshold:.1f} months")
        
        return {
            'hsc_status': hsc_status,
            'telomere_status': telomere_status,
            'current_hsc': current_hsc,
            'current_telomere': current_telomere,
            'hsc_analysis': hsc_analysis,
            'telomere_analysis': telomere_analysis,
            'trigger_second_page': trigger_second_page,
            'trigger_reasons': trigger_reasons,
            'time_to_hsc_threshold': time_to_hsc_threshold,
            'time_to_telomere_threshold': time_to_telomere_threshold,
            'overall_risk': self._determine_overall_risk(hsc_status, telomere_status, trigger_second_page)
        }
    
    def _get_hsc_status(self, value: float) -> str:
        """Determine HSC status based on value"""
        if value >= self.hsc_healthy_threshold:
            return "Healthy"
        elif value >= self.hsc_concern_threshold:
            return "Borderline" 
        else:
            return "High Concern"
    
    def _get_telomere_status(self, value: float) -> str:
        """Determine telomere status based on percentile"""
        if value >= self.telomere_healthy_threshold:
            return "Healthy"
        elif value >= self.telomere_concern_threshold:
            return "Borderline"
        else:
            return "High Concern"
    
    def _calculate_time_to_threshold(self, slope: float, intercept: float, current_time: float, threshold: float) -> float:
        """Calculate time to reach threshold using linear regression"""
        if slope >= 0:
            return float('inf')  # Not declining
        
        # Time Left = (Threshold - b) / m - Current Time Passed
        time_to_threshold = (threshold - intercept) / slope
        time_left = time_to_threshold - current_time
        
        return max(0, time_left)
    
    def _determine_overall_risk(self, hsc_status: str, telomere_status: str, trigger_second_page: bool) -> str:
        """Determine overall patient risk level"""
        if trigger_second_page:
            if "High Concern" in [hsc_status, telomere_status]:
                return "Critical"
            else:
                return "High"
        elif "Borderline" in [hsc_status, telomere_status]:
            return "Moderate"
        else:
            return "Low"
    
    def generate_graphs(self, patient_data: Dict[str, Any], analysis_result: Dict[str, Any]) -> Dict[str, str]:
        """Generate base64-encoded graphs for the second page"""
        dates = patient_data['dates']
        hsc_values = patient_data['hsc_values']
        telomere_values = patient_data['telomere_values']
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # HSC Graph
        ax1.plot(dates, hsc_values, 'bo-', label='HSC % Measurements', linewidth=2, markersize=8)
        ax1.plot(dates, analysis_result['hsc_analysis']['trend_line'], 'r--', label='Trend Line', linewidth=2)
        ax1.axhline(y=self.hsc_healthy_threshold, color='green', linestyle='-', alpha=0.7, label='Healthy Threshold (1%)')
        ax1.axhline(y=self.hsc_concern_threshold, color='red', linestyle='-', alpha=0.7, label='Concern Threshold (0.5%)')
        
        # Highlight current point
        current_hsc = hsc_values[-1]
        color = 'red' if current_hsc < 0.5 else 'orange' if current_hsc < 1.0 else 'green'
        ax1.plot(dates[-1], current_hsc, 'o', color=color, markersize=12, markeredgecolor='black', markeredgewidth=2)
        
        ax1.set_title('Bone Marrow Stem Cell Percentage Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('HSC Percentage (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        # Telomere Graph
        ax2.plot(dates, telomere_values, 'go-', label='Telomere Length Percentile', linewidth=2, markersize=8)
        ax2.plot(dates, analysis_result['telomere_analysis']['trend_line'], 'r--', label='Trend Line', linewidth=2)
        ax2.axhline(y=self.telomere_healthy_threshold, color='green', linestyle='-', alpha=0.7, label='Healthy Threshold (10th %ile)')
        ax2.axhline(y=self.telomere_concern_threshold, color='red', linestyle='-', alpha=0.7, label='Concern Threshold (1st %ile)')
        
        # Highlight current point
        current_telomere = telomere_values[-1]
        color = 'red' if current_telomere < 1 else 'orange' if current_telomere < 10 else 'green'
        ax2.plot(dates[-1], current_telomere, 'o', color=color, markersize=12, markeredgecolor='black', markeredgewidth=2)
        
        ax2.set_title('Telomere Length Percentile Over Time', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Age-Adjusted Percentile')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.tight_layout()
        
        # Convert to base64
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_base64


class ClinicalNLPProcessor:
    """Natural Language Processing for clinical notes"""
    
    def __init__(self):
        # Symptom keyword mappings for pattern detection
        self.symptom_keywords = {
            'fatigue': ['fatigue', 'tired', 'exhaustion', 'weakness', 'low energy', 'lethargy'],
            'bruising': ['bruising', 'bruises', 'contusions', 'ecchymoses', 'easy bruising'],
            'bleeding': ['bleeding', 'nosebleeds', 'epistaxis', 'hemorrhage', 'gum bleeding', 'petechiae'],
            'pallor': ['pallor', 'pale', 'paleness', 'ashen'],
            'petechiae': ['petechiae', 'pinpoint spots', 'small red spots'],
            'infections': ['infections', 'fever', 'recurrent illness', 'frequent colds'],
            'shortness_of_breath': ['shortness of breath', 'dyspnea', 'breathless', 'sob'],
            'weight_loss': ['weight loss', 'losing weight', 'unintentional weight loss'],
            'headaches': ['headaches', 'headache', 'head pain', 'cephalgia'],
            'lymphadenopathy': ['lymphadenopathy', 'swollen lymph nodes', 'enlarged nodes'],
            'night_sweats': ['night sweats', 'sweating at night', 'nocturnal sweating']
        }
        
        # Lab value patterns
        self.lab_patterns = {
            'low_wbc': ['low wbc', 'leukopenia', 'white blood cell.*low'],
            'low_platelets': ['low platelets', 'thrombocytopenia', 'platelet.*low'],
            'low_hemoglobin': ['low hemoglobin', 'anemia', 'hgb.*low', 'hb.*low'],
            'pancytopenia': ['pancytopenia', 'low counts', 'cytopenias']
        }
        
        # Aplastic Anemia diagnostic criteria
        self.aplastic_anemia_patterns = {
            'primary_symptoms': ['fatigue', 'bruising', 'bleeding', 'pallor'],
            'secondary_symptoms': ['petechiae', 'infections', 'shortness_of_breath'],
            'lab_findings': ['pancytopenia', 'low_wbc', 'low_platelets', 'low_hemoglobin']
        }
    
    def extract_symptoms(self, text: str) -> Dict[str, int]:
        """Extract symptoms from clinical text and count occurrences"""
        text_lower = text.lower()
        symptom_counts = {}
        
        for symptom, keywords in self.symptom_keywords.items():
            count = 0
            for keyword in keywords:
                # Use regex for more precise matching
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = re.findall(pattern, text_lower)
                count += len(matches)
            
            if count > 0:
                symptom_counts[symptom] = count
        
        return symptom_counts
    
    def extract_lab_findings(self, text: str) -> Dict[str, bool]:
        """Extract laboratory findings from clinical text"""
        text_lower = text.lower()
        lab_findings = {}
        
        for lab, patterns in self.lab_patterns.items():
            found = False
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    found = True
                    break
            lab_findings[lab] = found
        
        return lab_findings
    
    def calculate_temporal_patterns(self, visits: List[Dict]) -> Dict[str, List[int]]:
        """Calculate symptom patterns across time"""
        symptom_timeline = {}
        
        for visit_idx, visit in enumerate(visits):
            symptoms = self.extract_symptoms(visit['notes'])
            for symptom in symptoms:
                if symptom not in symptom_timeline:
                    symptom_timeline[symptom] = []
                symptom_timeline[symptom].append(visit_idx + 1)
        
        return symptom_timeline


class AplasticAnemiaDetector:
    """Specialized detector for Aplastic Anemia patterns"""
    
    def __init__(self, nlp_processor: ClinicalNLPProcessor):
        self.nlp = nlp_processor
    
    def assess_risk(self, visits: List[Dict]) -> Dict[str, Any]:
        """Assess risk of Aplastic Anemia based on visit patterns"""
        # Aggregate all symptoms across visits
        all_symptoms = {}
        all_labs = {}
        
        for visit in visits:
            symptoms = self.nlp.extract_symptoms(visit['notes'])
            labs = self.nlp.extract_lab_findings(visit['notes'])
            
            for symptom, count in symptoms.items():
                all_symptoms[symptom] = all_symptoms.get(symptom, 0) + count
            
            for lab, present in labs.items():
                if present:
                    all_labs[lab] = True
        
        # Calculate risk factors
        primary_count = sum(1 for symptom in self.nlp.aplastic_anemia_patterns['primary_symptoms'] 
                           if symptom in all_symptoms)
        
        secondary_count = sum(1 for symptom in self.nlp.aplastic_anemia_patterns['secondary_symptoms'] 
                             if symptom in all_symptoms)
        
        lab_count = sum(1 for lab in self.nlp.aplastic_anemia_patterns['lab_findings'] 
                       if all_labs.get(lab, False))
        
        # Check for symptom progression
        has_progression = len(visits) >= 3 and (
            all_symptoms.get('fatigue', 0) >= 2 or 
            all_symptoms.get('bruising', 0) >= 2
        )
        
        # Risk assessment logic
        if primary_count >= 3 and has_progression and lab_count >= 1:
            risk_level = 'high'
            message = ('Patient presents with classic triad of aplastic anemia symptoms '
                      '(fatigue, bruising, bleeding) with progression over multiple visits. '
                      'Laboratory abnormalities present. Immediate hematology consultation recommended.')
        elif primary_count >= 2 and (has_progression or secondary_count >= 1):
            risk_level = 'moderate'
            message = ('Patient shows concerning pattern with multiple primary symptoms '
                      'of aplastic anemia. Consider complete blood count with differential '
                      'and hematology consultation.')
        elif primary_count >= 1 and secondary_count >= 1:
            risk_level = 'low-moderate'
            message = ('Some symptoms consistent with aplastic anemia present. '
                      'Monitor closely and consider basic laboratory evaluation.')
        else:
            risk_level = 'low'
            message = ('Current symptom pattern does not strongly suggest aplastic anemia, '
                      'but continue routine monitoring.')
        
        return {
            'risk_level': risk_level,
            'message': message,
            'primary_symptoms_count': primary_count,
            'secondary_symptoms_count': secondary_count,
            'lab_findings_count': lab_count,
            'has_progression': has_progression,
            'symptoms_found': list(all_symptoms.keys()),
            'labs_found': [lab for lab, present in all_labs.items() if present]
        }


class RecommendationEngine:
    """Generate clinical recommendations based on analysis"""
    
    def __init__(self):
        self.research_articles = {
            'aplastic_anemia': [
                {
                    'title': 'Aplastic Anemia: Pathophysiology and Treatment - NEJM 2024',
                    'url': 'https://www.nejm.org/doi/full/10.1056/NEJMra1713802',
                    'summary': 'Comprehensive review of aplastic anemia pathophysiology and current treatment approaches'
                },
                {
                    'title': 'Early Detection of Aplastic Anemia: Clinical Guidelines - Blood Journal',
                    'url': 'https://ashpublications.org/blood/article/140/11/1190/485140/How-I-treat-aplastic-anemia',
                    'summary': 'Updated clinical guidelines for early detection and diagnosis'
                },
                {
                    'title': 'Aplastic Anemia - Mayo Clinic Clinical Overview',
                    'url': 'https://www.mayoclinic.org/diseases-conditions/aplastic-anemia/symptoms-causes/syc-20355015',
                    'summary': 'Clinical overview of aplastic anemia symptoms, causes, and diagnosis'
                },
                {
                    'title': 'Aplastic Anemia - National Heart, Lung, and Blood Institute',
                    'url': 'https://www.nhlbi.nih.gov/health/aplastic-anemia',
                    'summary': 'Comprehensive guide to aplastic anemia from NHLBI'
                }
            ],
            'hematology_general': [
                {
                    'title': 'AI in Hematology - Nature Medicine Reviews',
                    'url': 'https://www.nature.com/articles/s41591-021-01614-0',
                    'summary': 'Review of artificial intelligence applications in hematologic diagnosis'
                },
                {
                    'title': 'Rare Blood Disorders - MedlinePlus',
                    'url': 'https://medlineplus.gov/blooddisorders.html',
                    'summary': 'Overview of rare blood disorders and diagnostic approaches'
                },
                {
                    'title': 'Clinical Decision Support in Hematology - PubMed',
                    'url': 'https://pubmed.ncbi.nlm.nih.gov/35613431/',
                    'summary': 'Research on clinical decision support systems in hematology practice'
                }
            ]
        }
    
    def generate_recommendations(self, risk_assessment: Dict[str, Any], 
                               symptoms: Dict[str, int]) -> List[Dict[str, str]]:
        """Generate clinical recommendations based on risk assessment"""
        recommendations = []
        
        risk_level = risk_assessment['risk_level']
        
        if risk_level == 'high':
            recommendations.extend([
                {
                    'type': 'URGENT',
                    'priority': 1,
                    'text': 'Order STAT CBC with differential, comprehensive metabolic panel, and reticulocyte count'
                },
                {
                    'type': 'REFERRAL',
                    'priority': 1,
                    'text': 'Immediate hematology consultation for suspected aplastic anemia'
                },
                {
                    'type': 'MONITORING',
                    'priority': 1,
                    'text': 'Monitor for signs of infection, bleeding, and further clinical deterioration'
                },
                {
                    'type': 'PRECAUTIONS',
                    'priority': 2,
                    'text': 'Consider infection precautions and bleeding precautions pending lab results'
                }
            ])
        
        elif risk_level in ['moderate', 'low-moderate']:
            recommendations.extend([
                {
                    'type': 'LABS',
                    'priority': 1,
                    'text': 'Order CBC with differential within 24-48 hours'
                },
                {
                    'type': 'FOLLOW-UP',
                    'priority': 2,
                    'text': 'Schedule follow-up in 1-2 weeks to reassess symptoms'
                },
                {
                    'type': 'MONITORING',
                    'priority': 2,
                    'text': 'Monitor for worsening fatigue, new bruising, or bleeding'
                }
            ])
        
        # Symptom-specific recommendations
        if symptoms.get('fatigue', 0) >= 2:
            recommendations.append({
                'type': 'ASSESSMENT',
                'priority': 2,
                'text': 'Comprehensive fatigue workup including thyroid function, B12, folate levels'
            })
        
        if symptoms.get('bruising', 0) >= 1 and symptoms.get('bleeding', 0) >= 1:
            recommendations.append({
                'type': 'COAGULATION',
                'priority': 2,
                'text': 'Consider coagulation studies (PT/PTT/INR) and platelet function assessment'
            })
        
        if symptoms.get('infections', 0) >= 2:
            recommendations.append({
                'type': 'IMMUNOLOGY',
                'priority': 2,
                'text': 'Consider immunoglobulin levels and lymphocyte subset analysis'
            })
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])
        
        return recommendations
    
    def get_research_articles(self, risk_level: str) -> List[Dict[str, str]]:
        """Get relevant research articles based on risk level"""
        if risk_level in ['high', 'moderate']:
            return self.research_articles['aplastic_anemia']
        else:
            return self.research_articles['hematology_general']


class DatabaseManager:
    """Manage patient data storage and retrieval"""
    
    def __init__(self, db_path: str = 'clinical_assistant.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with patient and visit tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create patients table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                date_of_birth DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create visits table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                visit_date DATE NOT NULL,
                notes TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        ''')
        
        # Create analysis results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                risk_level TEXT,
                risk_message TEXT,
                symptoms_json TEXT,
                recommendations_json TEXT,
                FOREIGN KEY (patient_id) REFERENCES patients (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_bone_marrow_patients(self) -> List[Dict[str, Any]]:
        """Get all bone marrow analysis patients"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, age, patient_type FROM bone_marrow_patients')
        patients = [
            {'id': row[0], 'name': row[1], 'age': row[2], 'type': row[3]} 
            for row in cursor.fetchall()
        ]
        conn.close()
        return patients
        
        # Insert sample data
        self.insert_sample_data()
        
        # Insert bone marrow data
        self.insert_bone_marrow_data()
    
    def insert_bone_marrow_data(self):
        """Insert mock bone marrow analysis data"""
        # Create bone marrow patients table
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bone_marrow_patients (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age INTEGER,
                patient_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bone_marrow_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                reading_date DATE NOT NULL,
                hsc_percentage REAL,
                telomere_percentile REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (patient_id) REFERENCES bone_marrow_patients (id)
            )
        ''')
        
        # Sample bone marrow patients
        bone_marrow_patients = [
            {'id': 'bm_patient1', 'name': 'Alex Johnson', 'age': 45, 'type': 'healthy'},
            {'id': 'bm_patient2', 'name': 'Jordan Smith', 'age': 52, 'type': 'borderline_declining'},
            {'id': 'bm_patient3', 'name': 'Casey Williams', 'age': 38, 'type': 'high_concern'}
        ]
        
        for patient in bone_marrow_patients:
            cursor.execute('''
                INSERT OR REPLACE INTO bone_marrow_patients (id, name, age, patient_type) 
                VALUES (?, ?, ?, ?)
            ''', (patient['id'], patient['name'], patient['age'], patient['type']))
        
        conn.commit()
        conn.close()
    
    def insert_sample_data(self):
        """Insert sample patient data for demonstration"""
        sample_patients = [
            {
                'id': 'patient1',
                'name': 'Sarah Johnson',
                'visits': [
                    {
                        'date': '2024-01-15',
                        'notes': 'Patient reports persistent fatigue over the past 3 weeks. No fever. Mild headaches. Sleep patterns normal. No significant weight loss.'
                    },
                    {
                        'date': '2024-02-28',
                        'notes': 'Follow-up visit. Fatigue continues. Patient now reports easy bruising on arms and legs. No recent trauma. Petechiae noted on examination.'
                    },
                    {
                        'date': '2024-04-10',
                        'notes': 'Patient returns with worsening symptoms. Severe fatigue limiting daily activities. Increased bruising. Pallor noted. Patient reports feeling short of breath with minimal exertion.'
                    },
                    {
                        'date': '2024-05-22',
                        'notes': 'Patient presents with concerning symptoms. Pancytopenia suspected based on clinical presentation. Severe fatigue, extensive bruising, pallor, and frequent infections over past month.'
                    },
                    {
                        'date': '2024-06-15',
                        'notes': 'Lab results confirm severe pancytopenia. CBC shows: WBC 2.1, Hgb 7.2, Platelets 15,000. Patient referred to hematology for bone marrow biopsy evaluation.'
                    }
                ]
            },
            {
                'id': 'patient2',
                'name': 'Michael Chen',
                'visits': [
                    {
                        'date': '2024-03-05',
                        'notes': 'Routine check-up. Patient reports occasional fatigue but attributes to work stress. Vital signs normal. Physical exam unremarkable.'
                    },
                    {
                        'date': '2024-04-20',
                        'notes': 'Patient returns for persistent cough and fatigue. No fever. Chest X-ray ordered. Some mild bruising noted but patient reports recent fall.'
                    },
                    {
                        'date': '2024-06-10',
                        'notes': 'Follow-up for ongoing fatigue. Patient now reports weight loss of 10 lbs over 2 months. Night sweats present. Lymphadenopathy noted on examination.'
                    }
                ]
            },
            {
                'id': 'patient3',
                'name': 'Emma Rodriguez',
                'visits': [
                    {
                        'date': '2024-02-12',
                        'notes': 'Patient presents with recurrent nosebleeds over past week. No trauma. Patient reports fatigue but mild. No significant medical history.'
                    },
                    {
                        'date': '2024-03-25',
                        'notes': 'Return visit for persistent nosebleeds. Now reports easy bruising and fatigue worsening. Petechiae observed during examination. Patient concerned about symptoms.'
                    },
                    {
                        'date': '2024-05-08',
                        'notes': 'Patient returns with alarming symptoms. Severe fatigue, extensive bruising, pale appearance. Reports frequent infections. Gum bleeding noted. Urgent lab work ordered.'
                    }
                ]
            }
        ]
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for patient in sample_patients:
            # Insert patient
            cursor.execute('''
                INSERT OR REPLACE INTO patients (id, name) VALUES (?, ?)
            ''', (patient['id'], patient['name']))
            
            # Insert visits
            for visit in patient['visits']:
                cursor.execute('''
                    INSERT OR REPLACE INTO visits (patient_id, visit_date, notes) VALUES (?, ?, ?)
                ''', (patient['id'], visit['date'], visit['notes']))
        
        conn.commit()
        conn.close()
    
    def get_patient_data(self, patient_id: str) -> Dict[str, Any]:
        """Retrieve patient data with all visits"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get patient info
        cursor.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
        patient_row = cursor.fetchone()
        
        if not patient_row:
            conn.close()
            return None
        
        # Get visits
        cursor.execute('''
            SELECT visit_date, notes FROM visits 
            WHERE patient_id = ? ORDER BY visit_date
        ''', (patient_id,))
        visit_rows = cursor.fetchall()
        
        conn.close()
        
        return {
            'id': patient_row[0],
            'name': patient_row[1],
            'visits': [
                {'date': row[0], 'notes': row[1]} 
                for row in visit_rows
            ]
        }
    
    def save_analysis_result(self, patient_id: str, analysis_result: Dict[str, Any]):
        """Save analysis results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO analysis_results 
            (patient_id, risk_level, risk_message, symptoms_json, recommendations_json)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            patient_id,
            analysis_result['risk_assessment']['risk_level'],
            analysis_result['risk_assessment']['message'],
            json.dumps(analysis_result['symptoms']),
            json.dumps(analysis_result['recommendations'])
        ))
        
        conn.commit()
        conn.close()


# Initialize components
nlp_processor = ClinicalNLPProcessor()
aa_detector = AplasticAnemiaDetector(nlp_processor)
recommendation_engine = RecommendationEngine()
db_manager = DatabaseManager()
bone_marrow_analyzer = BoneMarrowAnalyzer()


# Flask Routes
@app.route('/')
def index():
    """Serve the main HTML interface"""
    return render_template('index.html')


@app.route('/api/patients', methods=['GET'])
def get_patients():
    """Get list of all patients"""
    conn = sqlite3.connect(db_manager.db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT id, name FROM patients')
    patients = [{'id': row[0], 'name': row[1]} for row in cursor.fetchall()]
    conn.close()
    
    return jsonify({'patients': patients})


@app.route('/api/patient/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    """Get specific patient data with visits"""
    patient_data = db_manager.get_patient_data(patient_id)
    
    if not patient_data:
        return jsonify({'error': 'Patient not found'}), 404
    
    return jsonify(patient_data)


@app.route('/api/analyze/<patient_id>', methods=['POST'])
def analyze_patient(patient_id):
    """Analyze patient for rare disease patterns"""
    patient_data = db_manager.get_patient_data(patient_id)
    
    if not patient_data:
        return jsonify({'error': 'Patient not found'}), 404
    
    # Perform analysis
    visits = patient_data['visits']
    
    # Extract symptoms across all visits
    all_symptoms = {}
    symptom_timeline = nlp_processor.calculate_temporal_patterns(visits)
    
    for visit in visits:
        symptoms = nlp_processor.extract_symptoms(visit['notes'])
        for symptom, count in symptoms.items():
            all_symptoms[symptom] = all_symptoms.get(symptom, 0) + count
    
    # Assess Aplastic Anemia risk
    risk_assessment = aa_detector.assess_risk(visits)
    
    # Generate recommendations
    recommendations = recommendation_engine.generate_recommendations(
        risk_assessment, all_symptoms
    )
    
    # Get research articles
    research_articles = recommendation_engine.get_research_articles(
        risk_assessment['risk_level']
    )
    
    # Prepare response
    analysis_result = {
        'patient_id': patient_id,
        'patient_name': patient_data['name'],
        'analysis_date': datetime.now().isoformat(),
        'visit_count': len(visits),
        'symptoms': all_symptoms,
        'symptom_timeline': symptom_timeline,
        'risk_assessment': risk_assessment,
        'recommendations': recommendations,
        'research_articles': research_articles
    }
    
    # Save analysis to database
    db_manager.save_analysis_result(patient_id, analysis_result)
    
    return jsonify(analysis_result)


@app.route('/api/prompt', methods=['POST'])
def analyze_prompt():
    """Analyze clinical prompt/query"""
    data = request.get_json()
    
    if not data or 'prompt' not in data:
        return jsonify({'error': 'Prompt text required'}), 400
    
    prompt = data['prompt']
    
    # Extract symptoms from prompt
    symptoms = nlp_processor.extract_symptoms(prompt)
    lab_findings = nlp_processor.extract_lab_findings(prompt)
    
    # Create mock visit for analysis
    mock_visits = [{'notes': prompt, 'date': datetime.now().strftime('%Y-%m-%d')}]
    
    # Assess risk
    risk_assessment = aa_detector.assess_risk(mock_visits)
    
    # Generate recommendations
    recommendations = recommendation_engine.generate_recommendations(
        risk_assessment, symptoms
    )
    
    # Get research articles
    research_articles = recommendation_engine.get_research_articles(
        risk_assessment['risk_level']
    )
    
    return jsonify({
        'prompt': prompt,
        'analysis_date': datetime.now().isoformat(),
        'symptoms_detected': symptoms,
        'lab_findings': lab_findings,
        'risk_assessment': risk_assessment,
        'recommendations': recommendations,
        'research_articles': research_articles
    })


@app.route('/bone-marrow')
def bone_marrow_index():
    """Bone marrow failure prediction interface"""
    return render_template('bone_marrow.html')


@app.route('/api/bone-marrow/patients', methods=['GET'])
def get_bone_marrow_patients():
    """Get all bone marrow patients"""
    patients = db_manager.get_bone_marrow_patients()
    return jsonify({'patients': patients})


@app.route('/api/bone-marrow/analyze/<patient_id>', methods=['POST'])
def analyze_bone_marrow_patient(patient_id):
    """Analyze bone marrow patient with prediction algorithms"""
    # Get patient info
    conn = sqlite3.connect(db_manager.db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT name, age, patient_type FROM bone_marrow_patients WHERE id = ?', (patient_id,))
    patient_row = cursor.fetchone()
    conn.close()
    
    if not patient_row:
        return jsonify({'error': 'Patient not found'}), 404
    
    patient_name, age, patient_type = patient_row
    
    # Generate mock data for this patient type
    patient_data = bone_marrow_analyzer.generate_mock_data(patient_type)
    
    # Perform risk assessment
    analysis_result = bone_marrow_analyzer.assess_patient_risk(patient_data)
    
    # Generate graphs if second page is triggered
    graph_data = None
    if analysis_result['trigger_second_page']:
        graph_data = bone_marrow_analyzer.generate_graphs(patient_data, analysis_result)
    
    # Prepare response
    response = {
        'patient_id': patient_id,
        'patient_name': patient_name,
        'patient_age': age,
        'patient_type': patient_type,
        'analysis_date': datetime.now().isoformat(),
        'current_hsc': analysis_result['current_hsc'],
        'current_telomere': analysis_result['current_telomere'],
        'hsc_status': analysis_result['hsc_status'],
        'telomere_status': analysis_result['telomere_status'],
        'overall_risk': analysis_result['overall_risk'],
        'trigger_second_page': analysis_result['trigger_second_page'],
        'trigger_reasons': analysis_result['trigger_reasons'],
        'time_to_hsc_threshold': analysis_result['time_to_hsc_threshold'],
        'time_to_telomere_threshold': analysis_result['time_to_telomere_threshold'],
        'hsc_trend_slope': analysis_result['hsc_analysis']['slope'],
        'telomere_trend_slope': analysis_result['telomere_analysis']['slope'],
        'graph_data': graph_data,
        'recommendations': self._generate_bone_marrow_recommendations(analysis_result),
        'export_data': self._prepare_export_data(patient_data, analysis_result, patient_name)
       

    }
    
    return jsonify(response)


def _generate_bone_marrow_recommendations(analysis_result: Dict[str, Any]) -> List[str]:
    """Generate clinical recommendations for bone marrow analysis"""
    recommendations = []
    
    if analysis_result['trigger_second_page']:
        recommendations.extend([
            "Recommend additional diagnostics: CBC with differential, bone marrow biopsy, cytogenetic analysis, and telomere testing.",
            "Consider hematology/oncology consultation for further evaluation.",
            "Monitor for signs of bone marrow failure: fatigue, frequent infections, easy bruising or bleeding.",
            "Schedule follow-up appointments every 4-6 weeks to track biomarker trends."
        ])
        
        if analysis_result['overall_risk'] == 'Critical':
            recommendations.insert(0, "URGENT: Immediate hematology consultation recommended due to critical biomarker levels.")
    
    else:
        recommendations.extend([
            "Continue routine monitoring with follow-up testing in 2-3 months.",
            "Maintain awareness of symptoms: unusual fatigue, increased infections, or bleeding tendencies.",
            "No immediate intervention required based on current biomarker levels."
        ])
    
    return recommendations


def _prepare_export_data(patient_data: Dict[str, Any], analysis_result: Dict[str, Any], patient_name: str) -> Dict[str, Any]:
    """Prepare data for export functionality"""
    return {
        'patient_name': patient_name,
        'export_timestamp': datetime.now().isoformat(),
        'biomarker_history': {
            'dates': [d.strftime('%Y-%m-%d') for d in patient_data['dates']],
            'hsc_values': patient_data['hsc_values'],
            'telomere_values': patient_data['telomere_values']
        },
        'current_status': {
            'hsc_status': analysis_result['hsc_status'],
            'telomere_status': analysis_result['telomere_status'],
            'overall_risk': analysis_result['overall_risk']
        },
        'prediction_metrics': {
            'hsc_trend_slope': analysis_result['hsc_analysis']['slope'],
            'telomere_trend_slope': analysis_result['telomere_analysis']['slope'],
            'time_to_hsc_threshold': analysis_result['time_to_hsc_threshold'],
            'time_to_telomere_threshold': analysis_result['time_to_telomere_threshold']
        },
        'trigger_indicators': analysis_result['trigger_reasons']
    }


if __name__ == '__main__':
    print("🏥 Clinical Assistant for Rare Disease Detection")
    print("🚀 Starting Flask application...")
    print()
    print("📋 Web Interface:")
    print("   🌐 Main App: http://localhost:5000/")
    print()
    print("🔬 API Endpoints:")
    print("   📊 List Patients:    GET  http://localhost:5000/api/patients")
    print("   👤 Patient Data:     GET  http://localhost:5000/api/patient/<id>")
    print("   🧬 Analyze Patient:  POST http://localhost:5000/api/analyze/<id>")
    print("   💬 Quick Analysis:   POST http://localhost:5000/api/prompt")
    print()
    print("📝 Example API Usage:")
    print("   curl -X GET http://localhost:5000/api/patients")
    print("   curl -X POST http://localhost:5000/api/analyze/patient1")
    print("   curl -X POST -H 'Content-Type: application/json' \\")
    print("        -d '{\"prompt\":\"fatigue and bruising\"}' \\")
    print("        http://localhost:5000/api/prompt")
    print()
    print("⚡ Ready for clinical analysis!")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)