import pandas as pd
import joblib
import streamlit as st

model = joblib.load("model.joblib")      # hakuna retraining
rules = model["rules"]

def mapendekezo(items, n=5):
    items = {i.lower().strip() for i in items}
    # chukua rules ambazo antecedents zote ziko kwenye items alizo nazo
    chaguo = rules[rules["antecedents"].apply(lambda a: a.issubset(items))]
    chaguo = chaguo[~chaguo["consequents"].apply(lambda c: c.issubset(items))]
    if chaguo.empty:
        return pd.DataFrame()
    chaguo = chaguo.sort_values("lift", ascending=False).head(n).copy()
    chaguo["antecedents"] = chaguo["antecedents"].apply(lambda s: ", ".join(s))
    chaguo["consequents"] = chaguo["consequents"].apply(lambda s: ", ".join(s))
    return chaguo[["antecedents", "consequents", "support", "confidence", "lift"]]

st.title("Grocery Recommendations")

# ---- Sidebar 1: Manual Entry ----
st.sidebar.header("Grocery Manual Entry")
item = st.sidebar.selectbox("Chagua item", model["items"])
if st.sidebar.button("Pata mapendekezo"):
    st.write("Items zinazohusiana na:", item)
    st.dataframe(mapendekezo([item]))

# ---- Sidebar 2: Upload Form ----
st.sidebar.header("Grocery Upload Form")
with st.sidebar.form("upload"):
    faili = st.file_uploader("Pakia CSV", type="csv")
    tuma = st.form_submit_button("Changanua")

if tuma:
    if faili is None:
        st.warning("Tafadhali pakia faili kwanza.")
    else:
        data = pd.read_csv(faili)
        data["itemDescription"] = data["itemDescription"].str.lower().str.strip()

        # kila basket (mteja + tarehe) inapata mapendekezo yake
        baskets = data.groupby(["Member_number", "Date"])["itemDescription"].apply(set)

        matokeo = []
        for (mteja, tarehe), items in baskets.head(200).items():   # 200 za kwanza tu
            rec = mapendekezo(items)
            matokeo.append({
                "Member_number": mteja,
                "Date": tarehe,
                "Items": ", ".join(sorted(items)),
                "Mapendekezo": ", ".join(rec["consequents"].unique()) if len(rec) else "-",
            })

        st.write("Mapendekezo kwa kila basket (200 za kwanza):")
        st.dataframe(pd.DataFrame(matokeo))    

