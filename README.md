

# Species Distribution Modeling (SDM) of Mountain Lion (Puma concolor) in the USA

> > Overview

Species Distribution Modeling (SDM) predicts the geographic distribution of species based on occurrence data and environmental variables. Accurate SDMs support conservation planning, habitat management, and ecological research.

This project demonstrates a hybrid **data-driven SDM workflow** for the Mountain Lion (*Puma concolor*) across the mainland USA. Occurrence data and environmental layers are used to train machine learning models that predict habitat suitability.

The work combines ecological modeling with modern deep learning techniques to generate habitat suitability maps and identify potential distribution areas.



> > Objectives

* Collect and preprocess occurrence data of *Puma concolor*
* Compile environmental layers (temperature, precipitation, elevation, land cover)
* Train machine learning models to predict habitat suitability
* Evaluate model performance under varying environmental conditions
* Generate geospatial maps of species distribution


> > Background

Species distribution models estimate where species are likely to occur by linking occurrence points with environmental features.

Environmental variability, climate change, and habitat loss make accurate modeling crucial for conservation. Traditional SDM methods like MaxEnt or GLMs may not capture complex nonlinear patterns. Machine learning provides a flexible alternative by learning habitat suitability directly from data.

This project integrates:

* Species Occurrence Data
* Environmental Layer Processing
* Machine Learning Classification & Regression
* Geospatial Visualization of Habitat Suitability


> > Technologies & Tools

* Python
* Pandas / NumPy
* Scikit-learn (Random Forest, Gradient Boosting)
* Matplotlib / Seaborn
* GeoPandas / Rasterio



> > System Architecture

```text
Species Occurrence Data
          ↓
Environmental Data Compilation
          ↓
Feature Engineering
          ↓
Machine Learning Model Training
          ↓
Habitat Suitability Prediction
          ↓
Geospatial Visualization

> > Methodology

1. Data Collection
   Occurrence records of *Puma concolor* are collected from GBIF and other biodiversity databases.

2. Environmental Data Preparation
   Temperature, precipitation, elevation, and land cover layers are collected, cleaned, and standardized.

3. Feature Engineering
   Environmental layers are combined with occurrence data to create feature matrices for model training.

4. Model Training
   Supervised learning models are trained to predict species presence based on environmental predictors.

5. Prediction & Visualization
   Trained models generate habitat suitability maps, highlighting regions where Mountain Lions are likely to occur.


> > Results

The SDM framework demonstrates:

* Accurate prediction of Mountain Lion habitat suitability
* Identification of high-probability distribution areas
* Reduced computational complexity compared to manual ecological assessments
* Geospatial maps for conservation and research applications



> > Applications

* Wildlife conservation and management
* Habitat suitability assessment
* Ecological research and biodiversity monitoring
* Climate change impact modeling on species distribution
* Environmental policy planning



> > Future Work

* Extend modeling to multi-species SDMs
* Incorporate temporal environmental data for seasonal modeling
* Integrate citizen science and real-time occurrence data
* Apply deep learning for spatial-temporal habitat prediction
* Combine with ecological connectivity and movement modeling

---

> > How to Run

bash
pip install pandas numpy scikit-learn matplotlib seaborn geopandas rasterio
python sdm_mountain_lion.py


> > Key Concepts

* Species Distribution Modeling (SDM)
* Habitat Suitability Prediction
* Environmental Feature Engineering
* Machine Learning Regression/Classification
* Geospatial Visualization


> > Author

**Srikanth Shanmugam**
Electronics & Instrumentation Engineer
AI • Ecology Modeling • Intelligent Scientific Systems

GitHub: [https://github.com/srikth](https://github.com/srikth)
LinkedIn: [https://www.linkedin.com/in/srikanth-shanmugam](https://www.linkedin.com/in/srikanth-shanmugam)



> > References

* Elith, J. & Leathwick, J. — Species Distribution Models: Ecological Explanation and Prediction
* Phillips, S. — MaxEnt Modeling for Species Habitat Prediction
* GBIF Occurrence Data for Puma concolor
* Scikit-learn Documentation (Python ML Library)
* Recent Research on Machine Learning for SDM

