import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# تحميل البيانات
df = pd.read_csv('D:\\programming project\\python\\data analsis\\san fransicco salary\\Salaries.csv')

# تنظيف البيانات
df['BasePay'] = pd.to_numeric(df['BasePay'], errors='coerce')
df['OvertimePay'] = pd.to_numeric(df['OvertimePay'], errors='coerce')
df['OtherPay'] = pd.to_numeric(df['OtherPay'], errors='coerce')
df['Benefits'] = pd.to_numeric(df['Benefits'], errors='coerce')
df['TotalPay'] = pd.to_numeric(df['TotalPay'], errors='coerce')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

# تخصيص مظهر الداشبورد
st.set_page_config(page_title="Salary Analysis Dashboard", layout="wide")

# استخدام CSS لجعل الخلفية سوداء والنصوص ذهبية
st.markdown("""
    <style>
    body {
        background-color: #1e1e1e;
        color: #d4af37;
    }
    h1, h2, h3 {
        color: #d4af37;
    }
    .css-1f6lgbv {
        color: #d4af37 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# إضافة لوجو
st.image('D:\\programming project\\python\\kivy\\kivyproject\\lo.png', width=200)  # ضع مسار الصورة الصحيح هنا

# عنوان الداشبورد
st.title('Comprehensive Salary Analysis')

# إضافة أدوات تحكم للمستخدم في الشريط الجانبي
st.sidebar.header("Filters")

# اختيار العام
selected_year = st.sidebar.selectbox('Select Year', options=df['Year'].unique(), index=len(df['Year'].unique()) - 1)

# اختيار وظيفة
selected_job = st.sidebar.selectbox('Select Job Title', options=df['JobTitle'].unique(), index=0)

# فلترة البيانات بناءً على اختيارات المستخدم
df_filtered_year = df[df['Year'] == selected_year]
df_filtered_job = df[df['JobTitle'] == selected_job]

# 1. تحليل الفروقات بين الوظائف
st.header("Salary Comparison by Job Title")
avg_salaries = df_filtered_year.groupby('JobTitle')['BasePay'].mean().reset_index().sort_values(by='BasePay', ascending=False)
st.write(avg_salaries)

# رسم بياني لمتوسط الرواتب حسب الوظيفة
st.subheader("Top 10 Jobs by Base Salary")
plt.figure(figsize=(10, 6))
sns.barplot(x='BasePay', y='JobTitle', data=avg_salaries.head(10), palette="viridis")
plt.title('Top 10 Jobs by Base Salary')
plt.xlabel("Average Base Salary")
plt.ylabel("Job Title")
st.pyplot(plt)

# 2. تحليل الرواتب الإضافية (OvertimePay و OtherPay)
st.header("Overtime and Other Pay Analysis")
avg_overtime = df_filtered_year.groupby('JobTitle')['OvertimePay'].mean().reset_index().sort_values(by='OvertimePay', ascending=False)
st.write("Top 5 Jobs by Overtime Pay:")
st.write(avg_overtime.head(5))

avg_otherpay = df_filtered_year.groupby('JobTitle')['OtherPay'].mean().reset_index().sort_values(by='OtherPay', ascending=False)
st.write("Top 5 Jobs by Other Pay:")
st.write(avg_otherpay.head(5))

# 3. تحليل الاتجاهات الزمنية
st.header("Salary Trends Over the Years")
avg_salaries_yearly = df.groupby('Year')['BasePay'].mean().reset_index()
st.line_chart(avg_salaries_yearly.set_index('Year'))

# 4. توزيع الرواتب
st.header("Salary Distribution")
plt.figure(figsize=(10, 6))
sns.histplot(df_filtered_year['BasePay'], bins=50, kde=True, color="purple")
plt.title('Salary Distribution')
plt.xlabel("Base Salary")
st.pyplot(plt)

# 5. التنبؤ بالرواتب المستقبلية (باستخدام الانحدار الخطي)
st.header("Future Salary Prediction")
df_year_salary = df.groupby('Year')['BasePay'].mean().reset_index()
X = df_year_salary['Year'].values.reshape(-1, 1)
y = df_year_salary['BasePay'].values

# بناء النموذج
model = LinearRegression()
model.fit(X, y)

# التنبؤ بالسنوات القادمة
years_future = np.array([2024, 2025, 2026]).reshape(-1, 1)
salary_pred = model.predict(years_future)

# عرض النتائج
pred_df = pd.DataFrame({'Year': [2024, 2025, 2026], 'Predicted Salary': salary_pred})
st.write("Salary Predictions for Upcoming Years:")
st.write(pred_df)

# 6. تحليل عدد الوظائف بمرور الوقت
st.header("Number of Jobs Over Time")
job_counts_per_year = df.groupby('Year')['JobTitle'].count().reset_index()
st.bar_chart(job_counts_per_year.set_index('Year'))

# 7. العلاقة بين الرواتب والمزايا
st.header("Base Salary vs Benefits")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='BasePay', y='Benefits', data=df_filtered_year, color="orange")
plt.title('Base Salary vs Benefits')
plt.xlabel("Base Salary")
plt.ylabel("Benefits")
st.pyplot(plt)

# 8. احتمالية الطلب المستقبلي والزيادة في الراتب
st.header("Job Title Future Demand and Salary Probability")

# احصائيات افتراضية لاحتمالية الطلب
job_demand_prob = {
    'Web Developer': 0.8,
    'Mobile Developer': 0.7,
    'Data Analyst': 0.6,
    'AI Specialist': 0.9,
    'Cybersecurity Specialist': 0.85
    # أضف المزيد كما يلزم
}

# حساب احتمالية الطلب
demand_prob = job_demand_prob.get(selected_job, 0.5)  # الافتراضي هو 0.5 إذا لم تكن الوظيفة موجودة في القاموس
st.write(f"Probability of Increased Demand for {selected_job}: {demand_prob * 100:.2f}%")

plt.figure(figsize=(10, 6))
plt.bar(['Current', 'Future'], [1 - demand_prob, demand_prob], color=['red', 'green'])
plt.title(f"Future Demand Probability for {selected_job}")
plt.ylabel("Probability")
st.pyplot(plt)
