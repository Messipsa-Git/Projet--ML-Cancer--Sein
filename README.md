# projet_ml_data_science_fast.py
# Version corrigée et testée

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           RocCurveDisplay, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BreastCancerClassifier:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        
    def load_data(self):
        """Charge et prépare les données"""
        print("Chargement des données...")
        data = load_breast_cancer()
        self.X = pd.DataFrame(data.data, columns=data.feature_names)
        self.y = pd.Series(data.target, name='target')
        
        # Informations sur les données
        print(f"Dimensions des données: {self.X.shape}")
        print(f"Nombre de classes: {len(np.unique(self.y))}")
        print(f"Distribution des classes:\n{self.y.value_counts()}")
        return self.X, self.y
    
    def split_data(self, test_size=0.2):
        """Divise les données en ensembles d'entraînement et de test"""
        print("\nDivision des données...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=self.random_state, stratify=self.y
        )
        print(f"Train set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def preprocess_data(self):
        """Prétraite les données"""
        print("\nPrétraitement des données...")
        # Standardisation pour la régression logistique
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Conversion en DataFrame pour conserver les noms de colonnes
        self.X_train_scaled = pd.DataFrame(self.X_train_scaled, columns=self.X.columns)
        self.X_test_scaled = pd.DataFrame(self.X_test_scaled, columns=self.X.columns)
        
        return self.X_train_scaled, self.X_test_scaled
    
    def initialize_models(self):
        """Initialise les modèles"""
        print("\nInitialisation des modèles...")
        self.models = {
            'LogisticRegression': LogisticRegression(max_iter=5000, random_state=self.random_state),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        }
        return self.models
    
    def cross_validation(self, cv=5):
        """Effectue une validation croisée"""
        print("\nValidation croisée...")
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            if name == 'LogisticRegression':
                X_cv = self.X_train_scaled
            else:
                X_cv = self.X_train
                
            scores = cross_val_score(model, X_cv, self.y_train, cv=skf, scoring='f1')
            print(f"CV F1 {name}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    def train_models(self):
        """Entraîne tous les modèles"""
        print("\nEntraînement des modèles...")
        for name, model in self.models.items():
            if name == 'LogisticRegression':
                X_train = self.X_train_scaled
            else:
                X_train = self.X_train
                
            model.fit(X_train, self.y_train)
            print(f"✅ {name} entraîné")
    
    def evaluate_model(self, model, X_test, y_test, name):
        """Évalue un modèle unique"""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"\n--- {name} ---")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Recall: {metrics['recall']:.3f}")
        print(f"F1-Score: {metrics['f1']:.3f}")
        print(f"ROC AUC: {metrics['roc_auc']:.3f}")
        print("Matrice de confusion:")
        print(metrics['confusion_matrix'])
        
        return metrics
    
    def evaluate_all_models(self):
        """Évalue tous les modèles"""
        print("\nÉvaluation des modèles...")
        self.results = {}
        
        for name, model in self.models.items():
            if name == 'LogisticRegression':
                X_test = self.X_test_scaled
            else:
                X_test = self.X_test
                
            self.results[name] = self.evaluate_model(model, X_test, self.y_test, name)
        
        # Déterminer le meilleur modèle basé sur F1-score
        self.best_model_name = max(self.results.keys(), 
                                 key=lambda x: self.results[x]['f1'])
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🌟 Meilleur modèle: {self.best_model_name} (F1: {self.results[self.best_model_name]['f1']:.3f})")
    
    def plot_roc_curves(self):
        """Trace les courbes ROC pour tous les modèles"""
        print("\nGénération des courbes ROC...")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for name, model in self.models.items():
            if name == 'LogisticRegression':
                X_test = self.X_test_scaled
            else:
                X_test = self.X_test
                
            y_prob = model.predict_proba(X_test)[:, 1]
            RocCurveDisplay.from_predictions(self.y_test, y_prob, name=name, ax=ax)
        
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Aléatoire')
        ax.set_title('Courbes ROC - Comparaison des modèles')
        ax.legend()
        plt.tight_layout()
        plt.savefig('roc_curves.png')  # Sauvegarde pour vérification
        plt.show()
    
    def plot_confusion_matrices(self):
        """Trace les matrices de confusion - VERSION CORRIGÉE"""
        print("\nGénération des matrices de confusion...")
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, metrics) in enumerate(self.results.items()):
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, 
                       annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Prédit 0', 'Prédit 1'],
                       yticklabels=['Vrai 0', 'Vrai 1'],
                       ax=axes[idx])
            axes[idx].set_title(f'Matrice de confusion - {name}')
            axes[idx].set_xlabel('Prédictions')
            axes[idx].set_ylabel('Vraies valeurs')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png')  # Sauvegarde pour vérification
        plt.show()
    
    def save_best_model(self, model_dir='models'):
        """Sauvegarde le meilleur modèle et les métriques - VERSION CORRIGÉE"""
        print("\nSauvegarde du modèle...")
        os.makedirs(model_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Préparer les données à sauvegarder
        model_data = {
            'model': self.best_model,
            'scaler': self.scaler,
            'features': list(self.X.columns),
            'best_model_name': self.best_model_name,
            'metrics': self.results[self.best_model_name],
            'timestamp': timestamp
        }
        
        # Sauvegarder le modèle
        model_path = os.path.join(model_dir, f'best_model_{timestamp}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Sauvegarder les métriques (version corrigée pour numpy arrays)
        metrics_path = os.path.join(model_dir, f'metrics_{timestamp}.json')
        json_metrics = {}
        for model_name, metrics in self.results.items():
            json_metrics[model_name] = {}
            for k, v in metrics.items():
                if k == 'confusion_matrix':
                    json_metrics[model_name][k] = v.tolist()  # Convertir numpy array en liste
                else:
                    json_metrics[model_name][k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
        
        with open(metrics_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"✅ Modèle sauvegardé: {model_path}")
        print(f"✅ Métriques sauvegardées: {metrics_path}")
        
        return model_path

def main():
    """Fonction principale"""
    print("🚀 Démarrage du projet de classification - Cancer du sein")
    
    # Initialiser le classifieur
    classifier = BreastCancerClassifier(random_state=42)
    
    try:
        # Charger et préparer les données
        classifier.load_data()
        classifier.split_data(test_size=0.2)
        classifier.preprocess_data()
        
        # Initialiser et entraîner les modèles
        classifier.initialize_models()
        classifier.cross_validation(cv=5)
        classifier.train_models()
        
        # Évaluer les modèles
        classifier.evaluate_all_models()
        
        # Visualisations
        classifier.plot_roc_curves()
        classifier.plot_confusion_matrices()
        
        # Sauvegarder le meilleur modèle
        classifier.save_best_model()
        
        print("\n✅ Projet terminé avec succès!")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()

# Version de test simplifiée
def test_rapide():
    """Version de test rapide sans les graphiques"""
    print("🧪 TEST RAPIDE - Démarrage...")
    
    classifier = BreastCancerClassifier(random_state=42)
    
    # Étapes essentielles seulement
    classifier.load_data()
    classifier.split_data(test_size=0.2)
    classifier.preprocess_data()
    classifier.initialize_models()
    classifier.train_models()
    classifier.evaluate_all_models()
    classifier.save_best_model()
    
    print("🧪 TEST RAPIDE - Terminé avec succès!")

if __name__ == "__main__":
    # Demander à l'utilisateur quel mode il veut
    print("Choisissez le mode d'exécution:")
    print("1 - Mode complet (avec graphiques)")
    print("2 - Mode test rapide (sans graphiques)")
    
    choix = input("Votre choix (1 ou 2): ").strip()
    
    if choix == "2":
        test_rapide()
    else:
        main()
