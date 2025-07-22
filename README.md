# ML-Assisted Identification of Transit Exoplanet Candidates

This project explores a Machine Learning (ML) approach for the automatic prediction of exoplanet candidates using data from NASA’s Transiting Exoplanet Survey Satellite (TESS).

The ML pipeline is designed to address the challenges of filtering noise and handling the increasing volume of data received from space missions targeting exoplanets orbiting Sun-like stars.

### Methods Used
The following ML algorithms were evaluated:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest

The models were trained on data from NASA’s ExoFOP-TESS database.

### Results
The **Random Forest** algorithm achieved the best performance, with an **F1 score of 0.974** on both scaled and non-scaled data.

### Future Work
This approach has the potential to be extended to other datasets from current and future space missions. Collaboration with professional astronomers could further refine and apply these techniques to support astronomical discovery.

---

### Tech Stack
- Python
- NumPy, Pandas
- Scikit-learn
- Matplotlib / Seaborn

---

### Project Structure
Key scripts and files include:
- `main.py`: Entry point for running the pipeline
- `preprocess.py`: Data cleaning and preprocessing
- `algorithms.py`: ML models and training
- `evaluation.py`: Evaluation metrics
- `*.csv`: Input and output datasets

---

### Contact
For questions or collaboration ideas, feel free to open an issue or reach out.
