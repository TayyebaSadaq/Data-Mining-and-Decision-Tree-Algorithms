# Data Mining and Descision Tree Algorithms
### Working with Patient appointment request dataset:

https://www.kaggle.com/datasets/carogonzalezgaltier/medical-appointment-scheduling-system

---
### Problem to be solved:

"Will a patient show up to their medical appointment?"

---
### Further Analysis - Regression or classification
For the aforementioned problem, I've gone with classification as we're predicting a binary outcome. Whether or not the patient will show up to their appointment - yes/no, true/false, 1/0.

Regression would be used if we were predicting a continuous value, like the number of days a patient has to wait in order to get an appointment or be seen by the doctor.

---
### EDA + Preprocessing
1. __Load the dataset__

    ![](images/image1.png)

2. __Check for missing values__

    for the logs where status is either "unattended" or "cancelled" the values for "check_in_time", "appointment_duration", "start_time", "end_time" and "waiting_time" are missing since obviously the appointment wasn't attended to begin with. however when checked specifically for attended appointment, there are no null values outside of the columns mentioned above.
<br>
    <ins>checking all columns for null values:</ins>

    ![checking all columns for null values](images/image2.png)
    <br>
    <ins>checking for null values in the "attended" status:</ins>

    ![checking null values for "attended" status](images/image3.png)

3. __Convert categorical data where needed (it was needed)__
    
    For this, status and sex will be converted to numerical values in order to be used for analysis and model building. Male = 0, Female = 1. Since status has more than 2 values, i'll be hot encoding it. creating new columns for each value and assigning 1 or 0 based on the value in the original column.

    <ins> removing male and female and replacing them with 0 and 1:</ins>

    ![replacing male and female with 0 and 1](images/image4.png)

    <ins> hot encoding the status column:</ins>

    Made column for each status type to then have numerical representation of whether it applied to the appointment and any null values for the appointment times etc upon cancellation/non attend is changeed to 0
    ![hot encoding the status column](images/image5.png)
    
4. __Removing noisy/irrelevant data__

    TBD yet - will be done after further analysis
5. __Data Visualisation__

    This is to get a better understanding of the data and see if there are any patters or trends to be observed. This will be done using: 

    a. Histogram - for distribution of data
    ![histogram](images/histograms.png)

    b. Boxplots - for outliers
    ![boxplot](images/boxplot.png)

    c. Correlation matrix - to see how the features are related to each other
    ![correlation matric](images/correlationmatrix.png)