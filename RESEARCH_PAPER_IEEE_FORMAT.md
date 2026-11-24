# Comparative Study of Mathematical, Machine Learning, and Deep Learning Models for Data Center Cooling Power Prediction

**Kavya Sharma**$^1$, **Anushka Singh**$^2$  
$^1$*Department of Computer Science and Engineering*  
$^2$*Department of Information Technology*  
*Email: {kavya.sharma, anushka.singh}@example.edu*

---

### Abstract
Cooling systems in modern data centers consume a large portion of total facility electricity, typically between 30% and 40%. Accurately predicting cooling power consumption helps data center operators schedule server workloads efficiently and reduce electricity costs. In this study, we present a straightforward comparative analysis of mathematical, machine learning (ML), and deep learning (DL) models to predict data center cooling power. We use a real-world multi-year dataset containing 73,696 continuous hourly records (2016–2025) of server IT power, outdoor temperature, outdoor humidity, and cooling power. We clean the data using hourly median filtering to remove sensor errors and split it chronologically into training (70%), validation (15%), and testing (15%) sets. We implement and evaluate nine different models: two mathematical baselines (Median baseline, Polynomial Regression), three standard ML algorithms (Random Forest with time features, Baseline LightGBM, Baseline XGBoost), one dynamic ML model (Dynamic GBDT with past cooling lags), and three deep learning models (LSTM, Tabular ResNet, and a Hybrid Ensemble). Experimental results show that standard ML models struggle with low accuracy ($R^2 \le 0.1911$), whereas deep learning models that consider past sequential history achieve the best performance. Specifically, the DL Hybrid Ensemble achieves an MAE of 2.48 kW, an RMSE of 3.82 kW, and an $R^2$ of 0.9195, reducing prediction error by 71.3% compared to baseline XGBoost. Statistical significance is confirmed using paired Wilcoxon signed-rank tests ($p < 0.001$). An ablation study reveals that server IT power is the most critical feature, followed by outdoor weather. Finally, cross-dataset validation on an unseen supercomputing facility (48,567 hours from the C3 data center) shows how models transfer to new environments and highlights the limitations of standard decision trees when server power exceeds training ranges.

### Index Terms
Data Center Cooling, Power Prediction, Regression Comparison, Machine Learning, Deep Learning, LSTM, Tabular ResNet, Ablation Study, Cross-Dataset Validation.

---

## I. Introduction

Data centers are the backbone of modern computing, hosting thousands of servers that power cloud platforms, web applications, and artificial intelligence models [1]. Running these computers generates a substantial amount of electrical power, almost all of which is converted into heat [2]. If this heat is not removed, computer chips overheat, throttle, and fail.

To keep servers at safe operating temperatures, data centers operate dedicated cooling systems composed of computer room air handlers, water chillers, pumps, and fans [3]. Cooling equipment accounts for **30% to 40% of the total electricity consumed by a data center** [4]. Because electricity costs represent a major operational expense, accurately forecasting cooling power demand is an essential practical problem [5].

If facility managers know how much cooling power will be needed in advance based on planned compute workloads and outdoor weather forecasts, they can:
1. Adjust chiller and fan settings proactively instead of reacting after temperatures rise.
2. Shift non-urgent background batch computing jobs to cooler nighttime hours to save electricity.
3. Prevent sudden peak electrical demand spikes that lead to utility penalties.

### Objective of this Paper
The simple goal of this project is to benchmark and compare different mathematical, machine learning, and deep learning models to predict data center cooling power:
* We collect and clean raw operational and weather data spanning 9.2 years (73,696 hourly records).
* We implement nine models across different families (baseline heuristics, tree-based machine learning, recurrent neural networks, and tabular deep learning).
* We perform statistical hypothesis testing (paired Wilcoxon signed-rank tests) to verify whether accuracy differences between models are statistically meaningful.
* We conduct a feature ablation study to measure how much server IT power, outdoor weather, and calendar time contribute to predictions.
* We perform cross-dataset validation by testing pre-trained models on an external, unseen supercomputer facility (48,567 records from the C3 data center at Idaho National Laboratory).

---

## II. Data Collection and Preprocessing

```
┌───────────────────────────┐      ┌───────────────────────────┐
│       power_file.csv      │      │      weather_file.csv     │
│ (Cooling kW, IT Power kW) │      │   (Outdoor Temp & Humid)  │
└─────────────┬─────────────┘      └─────────────┬─────────────┘
              │                                  │
              ▼                                  ▼
      Regex Timestamp Standardizer & UTC Time Conversion
              │                                  │
              └────────────────┬─────────────────┘
                               │
                               ▼
               Hourly Median Resampling & Spike Filter
                               │
                               ▼
                 Clean Merged Dataset (73,696 Hours)
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
     Training Set        Validation Set          Test Set
     (First 70%)           (Next 15%)          (Final 15%)
    51,587 records       11,054 records       11,055 records
   (2016 to Jan 2023)  (Jan 2023 - May 2024) (May 2024 - Aug 2025)
```
*Fig. 1. End-to-end data processing and chronological splitting pipeline.*

### A. Raw Data Sources
The primary experimental dataset comes from historical telemetry recorded at an operational data center:
1. `power_file.csv` (573.2 MB): Contains sub-hourly power meter readings for server power (`it_power_kw`) and cooling system power (`cooling_kw`).
2. `weather_file.csv` (371.5 MB): Contains concurrent local weather station readings for outdoor air temperature (`outdoor_air_temp`) in °F and relative humidity (`outdoor_air_humidity`) in %.

### B. Timestamp Parsing & Timezone Cleaning
Raw timestamps contained regional timezone naming strings in trailing parentheses:
$$\texttt{"Thu Jul 14 2016 09:15:00 GMT+0000 (Coordinated Universal Time)"}$$
We cleaned the strings using regular expressions to strip parenthetical text and converted all timestamps into standard UTC ISO-8601 datetime format.

### C. Resampling and Spike Filtering: Median vs. Mean
Raw electrical meters logged data at variable 1- to 15-minute intervals. Sensor dropouts and transient electrical glitches caused isolated measurement spikes exceeding 500 kW. 

To create clean hourly records, we compared **Hourly Mean** versus **Hourly Median** aggregation:
* **Hourly Mean**: Kept the transient sensor errors, producing an unrealistic maximum cooling power of $512.40\text{ kW}$.
* **Hourly Median**: Naturally ignored isolated pulse spikes and preserved the true operational steady-state, resulting in a realistic maximum cooling power of $62.09\text{ kW}$ and a clean standard deviation of $9.07\text{ kW}$.

Therefore, hourly median aggregation was used. The resulting clean dataset contains **73,696 continuous hourly records** spanning **June 12, 2016 to August 29, 2025** (9.2 years).

### D. Chronological Data Split
Because data center workloads and weather change over time, random train/test splitting causes data leakage. We split the dataset strictly in chronological order:
* **Training Set (70%)**: 51,587 records (June 12, 2016 to January 4, 2023). Used to train the models.
* **Validation Set (15%)**: 11,054 records (January 4, 2023 to May 13, 2024). Used for hyperparameter tuning and early stopping.
* **Test Set (15%)**: 11,055 records (May 14, 2024 to August 29, 2025). Held out strictly for final evaluation.

---

## III. Models Evaluated

We implemented and compared nine models categorized into four main families:

### A. Mathematical and Heuristic Baselines
1. **Median Baseline**: A naive heuristic that always predicts the median cooling power observed in the training set ($8.27\text{ kW}$). This serves as a sanity check; any useful model must outperform this baseline.
2. **Polynomial Regression (Degree 2)**: Fits a second-order polynomial curve on server IT power and weather features to capture basic non-linear curves.

### B. Standard Machine Learning Models
3. **Time Features Model (Random Forest)**: A Random Forest regressor trained on IT power, outdoor temperature, outdoor humidity, plus basic calendar time features (hour of day, day of week, month).
4. **Baseline LightGBM**: A fast gradient boosted decision tree model trained on current IT power, temperature, and humidity.
5. **Baseline XGBoost**: An extreme gradient boosting tree model trained on IT power, temperature, humidity, and calendar features with early stopping.

### C. Dynamic Machine Learning Model
6. **Dynamic Autoregressive GBDT**: Standard models only look at current conditions ($t$). However, buildings retain heat over time. We engineer past cooling lag features ($t-1, t-2, t-3, t-6, t-24$ hours) and rolling window statistics (rolling mean and standard deviation over 3, 6, 12, and 24 hours). This allows the tree model to account for recent cooling trends.

### D. Deep Learning Models
7. **DL LSTM (Long Short-Term Memory)**: A 2-layer recurrent neural network (64 hidden units, dropout = 0.10). It receives a 24-hour sliding window of historical server workloads and weather conditions, learning sequential time patterns and thermal delay.
8. **DL Tabular ResNet**: A deep neural network designed for tabular data. It projects 66 standardized features into 128 dimensions, passes them through two residual blocks with skip connections ($\mathbf{x} + F(\mathbf{x})$), Batch Normalization, GELU activation functions, and dropout (0.15), followed by a linear regression head. Skip connections help gradient flow and prevent performance degradation in deep layers.
9. **DL Hybrid Ensemble**: Combines the predictions of the Tabular ResNet and the Dynamic GBDT using weighted linear stacking:
   $$\hat{y}_{\text{Hybrid}}(t) = 0.80 \cdot \hat{y}_{\text{ResNet}}(t) + 0.20 \cdot \hat{y}_{\text{Dynamic\_GBDT}}(t)$$

---

## IV. Experimental Setup & Evaluation Metrics

### A. Evaluation Metrics
We measure prediction performance using three standard regression metrics:
* **Mean Absolute Error (MAE)**: Measures average prediction error magnitude in kilowatts:
  $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i| \quad (\text{kW})$$
* **Root Mean Squared Error (RMSE)**: Penalizes larger individual errors more heavily:
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2} \quad (\text{kW})$$
* **Coefficient of Determination ($R^2$)**: Measures the percentage of variance in cooling power explained by the model:
  $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

### B. Statistical Significance Testing
To confirm that model improvements are not due to random chance, we run the non-parametric **Paired Wilcoxon Signed-Rank Test** comparing the absolute errors of each model against the Median baseline:
$$W = \sum_{i=1}^{N_r} \left[ \text{sgn}(|e_{\text{median}, i}| - |e_{\text{model}, i}|) \cdot R_i \right]$$
We adjust all resulting $p$-values using the **Holm-Bonferroni correction** to control for multiple hypothesis testing.

---

## V. Results & Model Comparison

Table I presents the evaluation metrics on the held-out test set (11,055 hours).

TABLE I  
Performance Comparison Across All Evaluated Models on Test Set (11,055 Hours)

| Rank | Model Name | Model Family | MAE (kW) | RMSE (kW) | $R^2$ Score | Paired Error Diff vs Median | Holm-Adjusted $p$-value | Statistical Verdict |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 9 | Polynomial Regression | Mathematical | 10.5365 | 13.9797 | -0.0772 | -1.82 kW | $p = 0.0$ | Significantly Worse |
| 8 | LightGBM Baseline | Standard ML | 9.0779 | 14.0363 | -0.0860 | -0.56 kW | $p < 10^{-36}$ | Significantly Worse |
| 7 | Time Features (RF) | Standard ML | 8.8853 | 12.4926 | 0.1398 | -0.14 kW | $p < 10^{-114}$ | Worse than Median |
| 6 | Median Baseline | Heuristic | 8.7203 | 12.3609 | 0.1578 | Reference | Reference | Baseline Reference |
| 5 | XGBoost Baseline | Standard ML | 8.6324 | 12.1141 | 0.1911 | -0.11 kW | $p = 0.00013$ | Slightly Better |
| 4 | Dynamic Autoregressive GBDT | Dynamic ML | 3.2829 | 6.2565 | 0.7842 | +3.80 kW | $p < 10^{-300}$ | Substantial Improvement |
| 3 | DL LSTM Sequence Model | Deep Learning | 2.4974 | 4.3867 | 0.8939 | +4.21 kW | $p < 10^{-300}$ | High Accuracy |
| 2 | DL Tabular ResNet | Deep Learning | 2.5846 | 3.9852 | 0.9125 | +3.64 kW | $p < 10^{-300}$ | High Accuracy |
| **1** | **🏆 DL Hybrid Ensemble** | **Deep Learning** | **2.4767** | **3.8213** | **0.9195** | **+3.72 kW** | **$p < 10^{-300}$** | **Best Overall Performance** |

```
                       TEST SET MAE (LOWER IS BETTER)
Polynomial Reg.  ██████████████████████████████ 10.54 kW
LightGBM Base    █████████████████████████ 9.08 kW
Time Features    ████████████████████████ 8.89 kW
Median Baseline  ███████████████████████ 8.72 kW
XGBoost Base     ███████████████████████ 8.63 kW
Dynamic GBDT     █████████ 3.28 kW
DL LSTM          ███████ 2.50 kW
DL Tabular ResNet███████ 2.58 kW
DL Hybrid Ensemb ███████ 2.48 kW (71.3% Error Reduction)
```
*Fig. 2. Comparison of Mean Absolute Error across all models.*

### Discussion of Model Performance:
1. **Classical and Static Models Perform Poorly**:
   * Polynomial Regression ($R^2 = -0.0772$) and Baseline LightGBM ($R^2 = -0.0860$) performed worse than the simple Median baseline ($R^2 = 0.1578$). A negative $R^2$ indicates that predicting the simple historical median is more accurate than these models. This happens because server power in recent years increased beyond the levels seen during training, causing static models to extrapolate erratically.
   * Standard XGBoost achieved only modest accuracy ($R^2 = 0.1911$, $\text{MAE} = 8.63\text{ kW}$).
2. **Adding Dynamic Lags Cuts Error by More than Half**:
   * The Dynamic GBDT model achieved an MAE of $3.28\text{ kW}$ and an $R^2$ of $0.7842$. Simply providing past cooling lags and rolling window averages allowed the model to understand the current operating state of the cooling plant.
3. **Deep Learning Achieves State-of-the-Art Accuracy**:
   * The DL LSTM sequence model reached $R^2 = 0.8939$ ($\text{MAE} = 2.50\text{ kW}$) by learning temporal dependencies across 24-hour sliding windows.
   * The DL Tabular ResNet reached $R^2 = 0.9125$ ($\text{MAE} = 2.58\text{ kW}$) using residual connections that facilitate training on complex multi-feature inputs.
   * The **DL Hybrid Ensemble** achieved the highest accuracy overall: **$\text{MAE} = 2.48\text{ kW}$**, **$\text{RMSE} = 3.82\text{ kW}$**, and **$R^2 = 0.9195$**. This represents a **$71.3\%$ reduction in MAE** compared to standard XGBoost.
4. **Statistical Significance**:
   * The paired Wilcoxon test confirmed that the Dynamic GBDT, LSTM, ResNet, and Hybrid Ensemble were all statistically superior to the median baseline ($p < 10^{-300}$).

---

## VI. Ablation Study

An **ablation study** investigates how much each input feature group contributes to the final prediction. We trained the baseline XGBoost model using identical hyperparameters and data splits while systematically removing one group of features at a time.

TABLE II  
Ablation Study Results on XGBoost Model

| Experiment | Features Kept | Test MAE (kW) | Test RMSE (kW) | Test $R^2$ | Impact on Performance |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Full Model** | IT Power + Weather + Time | **8.6509** | **12.1210** | **0.1902** | Complete feature baseline. |
| **No Time Features** | IT Power + Weather | 8.6013 | 12.3158 | 0.1639 | Minor change ($\Delta R^2 = -0.026$). Diurnal cycles are largely captured by weather. |
| **No Weather Features**| IT Power + Time | 9.1462 | 13.1736 | 0.0434 | Large drop ($\Delta R^2 = -0.147$). Outdoor weather is necessary for predicting chiller effort. |
| **No IT Power** | Weather + Time | 9.0768 | 14.0365 | -0.0860 | **Complete failure ($\Delta R^2 = -0.276$). Model performs worse than the simple median.** |

### Key Takeaways from the Ablation Study:
1. **Server IT Power is the Single Most Critical Feature**:
   * When server IT power was removed, the model completely broke down ($R^2$ dropped from $+0.1902$ to $-0.0860$). Because servers generate the heat that cooling systems must remove, predicting cooling power without server load is impossible.
2. **Outdoor Weather Provides Crucial Environmental Context**:
   * Removing outdoor temperature and humidity caused $R^2$ to drop from $0.1902$ to $0.0434$. Ambient temperature dictates how hard cooling systems must work to reject heat outside.
3. **Calendar Time Features are Helpful but Secondary**:
   * Removing time features (hour, day, month) caused only a small drop in $R^2$ (from $0.1902$ to $0.1639$), showing that time features mostly serve as proxies for daily temperature fluctuations.

---

## VII. Cross-Dataset Validation

To evaluate how models perform when transferred to a completely new, unseen data center without retraining, we performed **cross-dataset validation** using telemetry from an independent supercomputing facility.

### A. The C3 Supercomputer Dataset
* **Facility**: Collaborative Computing Center (C3, Building IF-692) at Idaho National Laboratory (INL).
* **Data Volume**: 48,567 continuous hourly records (July 2020 to January 2026), with **38,469 synchronized hours** overlapping with on-site weather records.
* **Workload Characteristics**: Unlike the enterprise facility, the C3 data center runs scientific high-performance computing (HPC) jobs. Mean compute load is higher ($1,586.12\text{ kW}$ vs. $1,497.02\text{ kW}$), and median baseline load is 24.6% higher ($1,475.72\text{ kW}$ vs. $1,184.52\text{ kW}$) because HPC clusters run 24/7 without night shutdowns.

### B. Cross-Dataset Results and Decision Tree Saturation
We deployed pre-trained models zero-shot onto the 38,469 synchronized hours of the C3 facility. Table III summarizes the resulting predictions.

TABLE III  
Cross-Dataset Validation Results on External C3 Facility (38,469 Hours)

| Model Name | Predicted Mean (kW) | Predicted Std (kW) | Implied PUE | Behavior on Unseen Data Center |
| :--- | :---: | :---: | :---: | :--- |
| Dynamic Autoregressive GBDT | 4.38 | 0.98 | 1.0030 | Responsive to sequential lag updates |
| Baseline XGBoost | 12.67 | 4.65 | 1.0086 | Moderate variation across seasons |
| Baseline Random Forest | 12.57 | 4.60 | 1.0086 | Moderate variation across seasons |
| **Baseline LightGBM** | **10.40** | **0.11** | **1.0075** | **Completely flatlined (standard dev = 0.11 kW)** |

### C. The Tree Leaf Boundary Saturation Bug
Cross-dataset testing uncovered a critical limitation of standard decision trees:
* **The Problem**: Baseline LightGBM produced an invariant prediction of **$10.40\text{ kW}$** ($\text{std} = 0.11\text{ kW}$) across all 38,469 hours, failing to respond to changing server workloads or outdoor weather.
* **Why it Happened**: Decision trees make predictions by splitting features using threshold rules ($x \le \theta$). In the primary dataset used for training, mean server power was $1,082\text{ kW}$. In the C3 supercomputer dataset, server power was consistently higher, with 89% of hours exceeding $1,300\text{ kW}$. Because query server loads were higher than any threshold learned during training, all samples were funneled into the exact same outermost leaf node.
* **Lesson for Practice**: Standard decision tree models cannot extrapolate to larger facilities. When server hardware is upgraded or models are transferred to larger data centers, models must be retrained or normalized to prevent leaf saturation.

---

## VIII. Practical Recommendations for Data Centers

Based on our empirical findings across both facilities, we recommend:
1. **Always Include Recent Historical Lags**:
   * Data center temperatures change gradually due to thermal inertia. Models that include past 1- to 6-hour cooling lags achieve a **60% to 70% error reduction** over static models.
2. **Prioritize Deep Learning for Production BMS**:
   * Tabular ResNet and recurrent LSTM architectures handle continuous features and complex multi-feature interactions better than shallow models, achieving $R^2 \approx 0.92$.
3. **Use Median Aggregation for Sensor Preprocessing**:
   * Raw power meter streams frequently contain transient electrical spikes. Using hourly median aggregation filters sensor errors without distorting baseline load curves.
4. **Monitor for Workload Range Drift**:
   * Before deploying a trained model, verify that incoming server workloads stay within the minimum and maximum ranges seen during training to prevent tree saturation.

---

## IX. Conclusion

In this study, we carried out a straightforward comparative analysis of mathematical, machine learning, and deep learning models for data center cooling power prediction across 14.7 years of operational telemetry from two facilities.

Our conclusions are:
1. **Model Hierarchy**: Naive heuristics and static models (Polynomial Regression, LightGBM, XGBoost) yield poor accuracy ($R^2 \le 0.1911$). In contrast, deep learning architectures (Tabular ResNet, LSTM) and dynamic models that capture sequential history achieve high accuracy ($R^2 = 0.9195$, $\text{MAE} = 2.48\text{ kW}$), representing a **71.3% error reduction**.
2. **Feature Importance**: Our ablation study confirmed that server IT power is the most critical feature ($R^2$ collapses to $-0.0860$ without it), outdoor weather is the secondary driver, and calendar time is auxiliary.
3. **Cross-Facility Transfer**: Testing on an unseen supercomputing facility revealed that standard tree models suffer from boundary leaf saturation when server workloads exceed training ranges.

Future work will examine online adaptive learning and model predictive control to dynamically adjust cooling setpoints in real time.

---

### References

* [1] N. Jones, "How to stop data centres from gobbling up the world's electricity," *Nature*, vol. 561, no. 7722, pp. 163–166, Sep. 2018.
* [2] E. Masanet et al., "Recalibrating global data center energy-use estimates," *Science*, vol. 367, no. 6481, pp. 984–986, Feb. 2020.
* [3] ASHRAE TC 9.9, *Thermal Guidelines for Data Processing Environments*, 5th ed. Atlanta, GA, USA: ASHRAE, 2021.
* [4] H. Rong, H. Zhang, S. Xiao, C. Li, and C. Hu, "Optimizing energy consumption for data centers: A survey," *IEEE Access*, vol. 4, pp. 6741–6761, 2016.
* [5] The Green Grid, "Usage and Public Reporting Guidelines for Carbon Usage Effectiveness (CUE) and Water Usage Effectiveness (WUE)," Tech. Rep. 35, 2011.
* [6] R. Gao, X. Jing, and M. Li, "Digital twin-driven energy modeling and predictive maintenance for high-density data centers," *IEEE Trans. Autom. Sci. Eng.*, vol. 19, no. 3, pp. 1823–1835, Jul. 2022.
* [7] J. Gao, "Machine learning applications for data center optimization," Google White Paper, Mountain View, CA, USA, Tech. Rep., 2014.
* [8] R. Evans and J. Gao, "DeepMind AI reduces Google data centre cooling bill by 40%," DeepMind Blog, Jul. 2016.
* [9] P. Zhang, C. Hu, and Y. Song, "Regime shifts in building HVAC operational telemetry: Diagnostic detection and adaptive calibration," *Appl. Energy*, vol. 285, p. 116482, Mar. 2021.
* [10] S. S. S. R. Depuru, L. Wang, and V. Devabhaktuni, "Smart meters for power grid: Challenges, issues, advantages and status," *Renew. Sustain. Energy Rev.*, vol. 15, no. 6, pp. 2736–2742, 2011.
* [11] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.*, San Francisco, CA, USA, 2016, pp. 785–794.
* [12] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *Adv. Neural Inf. Process. Syst.*, Long Beach, CA, USA, 2017, pp. 3146–3154.
* [13] Y. Li, Y. Han, and T. Xu, "Data center room temperature prediction based on machine learning," in *Proc. IEEE 4th Int. Conf. Big Data Comput. Serv. Appl. (BigDataService)*, Bamberg, Germany, 2018, pp. 217–222.
* [14] L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?," in *Adv. Neural Inf. Process. Syst.*, New Orleans, LA, USA, 2022, pp. 507–520.
* [15] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Comput.*, vol. 9, no. 8, pp. 1735–1780, Nov. 1997.
* [16] Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko, "Revisiting deep learning models for tabular data," in *Adv. Neural Inf. Process. Syst.*, 2021, pp. 18932–18943.
* [17] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Las Vegas, NV, USA, 2016, pp. 770–778.
* [18] ASHRAE, *Guideline 14-2014: Measurement of Energy, Demand, and Water Savings*, Atlanta, GA, USA: ASHRAE, 2014.
* [19] S. J. Pan and Q. Yang, "A survey on transfer learning," *IEEE Trans. Knowl. Data Eng.*, vol. 22, no. 10, pp. 1345–1359, Oct. 2010.
