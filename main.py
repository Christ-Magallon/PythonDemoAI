from sqlalchemy import create_engine
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

#Fetch Purchase History & Product Categories
def fetch_data():
    engine = create_engine("mysql+mysqlconnector://root:magallon123@localhost:3306/DemoDb")
    query = """
    SELECT CAST(ph.UserId AS CHAR) AS UserId, 
           ph.ProductId, 
           p.category, 
           SUM(ph.Quantity) AS PurchaseCount
    FROM PurchaseHistories ph
    JOIN Products p ON ph.ProductId = p.id
    GROUP BY ph.UserId, ph.ProductId, p.category;
    """
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    
    df["UserId"] = df["UserId"].astype(str)

    scaler = MinMaxScaler()
    df["PurchaseCount"] = scaler.fit_transform(df[["PurchaseCount"]])

    return df

#Convert Data into Pivot Table
def preprocess_data(df):
    return df.pivot_table(index="UserId", columns="ProductId", values="PurchaseCount", fill_value=0)

#Train KNN Model on Users
def train_model(df_pivot):
    model = NearestNeighbors(metric="cosine", algorithm="brute")
    model.fit(df_pivot) 
    return model

#Recommend Products Based on Similarity
def recommend_products(user_id, model, df_pivot, df_raw, n_recommendations=5):
    user_id = str(user_id)
    
    if user_id not in df_pivot.index:
        return []  

    _, indices = model.kneighbors(df_pivot.loc[[user_id]].values.reshape(1, -1), n_neighbors=6)  

    similar_users = df_pivot.iloc[indices[0][1:]].index.tolist()  
    similar_users_purchases = df_pivot.loc[similar_users].sum(axis=0)

    target_user_purchases = df_pivot.loc[user_id]
    unpurchased_products = similar_users_purchases[target_user_purchases == 0]

    user_categories = df_raw[df_raw["UserId"] == user_id]["category"].unique()
    df_products = df_raw[["ProductId", "category"]].drop_duplicates()
    df_products = df_products.set_index("ProductId")  
    prioritized_products = unpurchased_products.index[
        unpurchased_products.index.map(lambda pid: df_products.loc[pid, "category"] in user_categories)
    ]

    if len(prioritized_products) >= n_recommendations:
        recommended_products = prioritized_products[:n_recommendations].tolist()
    else:
        other_products = unpurchased_products.index.difference(prioritized_products)[:n_recommendations - len(prioritized_products)]
        recommended_products = prioritized_products.tolist() + other_products.tolist()

    return recommended_products