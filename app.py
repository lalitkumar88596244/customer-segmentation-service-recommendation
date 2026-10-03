import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Analytics", page_icon="📊", layout="wide")
st.title("Customer Segmentation & Service Recommendation System")
st.caption("Veda Technology | Major Project — Data Science Track")
up=st.sidebar.file_uploader("Upload customer CSV (optional)",type=["csv"])
data=pd.read_csv(up) if up else pd.read_csv("customer_data.csv")
features=["website_visits","engagement_score","monthly_spend","service_inquiries","support_requests","training_interest"]
data=data.copy()
for c in features: data[c]=pd.to_numeric(data[c],errors="coerce")
data[features]=data[features].fillna(data[features].median())
if "customer_id" not in data: data["customer_id"]=[f"CUST{i+1:04d}" for i in range(len(data))]
k=st.sidebar.slider("Number of segments (K)",2,8,4)
X=StandardScaler().fit_transform(data[features])
data["cluster"]=KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(X)
score=silhouette_score(X,data.cluster) if len(data)>k else 0
a,b,c=st.columns(3); a.metric("Customers",len(data)); b.metric("Segments",k); c.metric("Silhouette score",f"{score:.3f}")
st.info("Default dataset is synthetic. Do not upload confidential customer information.")
st.subheader("Segment distribution")
fig,ax=plt.subplots(); data.cluster.value_counts().sort_index().plot(kind="bar",ax=ax); ax.set_xlabel("Cluster"); ax.set_ylabel("Customers"); st.pyplot(fig)
st.subheader("PCA cluster visualization")
p=PCA(n_components=2,random_state=42).fit_transform(X); fig,ax=plt.subplots()
for cl in sorted(data.cluster.unique()):
    ix=data.cluster==cl; ax.scatter(p[ix,0],p[ix,1],label=f"Segment {cl}",alpha=.7)
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(); st.pyplot(fig)
st.subheader("Segment profiles"); st.dataframe(data.groupby("cluster")[features].mean().round(2),use_container_width=True)
def recommend(r):
    if r.monthly_spend>=data.monthly_spend.quantile(.75) and r.engagement_score>=4: return "Premium IT services, advanced training, priority support"
    if r.training_interest>=4: return "Career training, workshops, learning programs"
    if r.service_inquiries>=5 or r.support_requests>=4: return "IT consultation, managed services, technical support"
    if r.website_visits<=data.website_visits.median(): return "Introductory webinar, beginner course, onboarding"
    return "Relevant short courses, service consultation, newsletters"
data["recommendation"]=data.apply(recommend,axis=1)
st.subheader("Customer recommendations"); st.dataframe(data[["customer_id","cluster","recommendation"]].head(50),use_container_width=True)
st.download_button("Download results CSV",data.to_csv(index=False).encode(),"customer_segmentation_results.csv","text/csv")
st.caption("Review segment averages before assigning business labels. Recommendations are illustrative rules and require validation.")
