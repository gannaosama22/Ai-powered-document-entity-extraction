import pandas as pd

d = pd.read_csv("results/document_summary.csv")
p = pd.read_csv("results/predictions.csv")

print("document_summary columns:", d.columns.tolist())
print("predictions columns:", p.columns.tolist())

# العمود المشترك بين الملفين (رقم الإيميل)
common = [c for c in p.columns if c in d.columns]
if not common:
    raise SystemExit("مفيش عمود مشترك بين الملفين، ابعتي أسماء الأعمدة اللي فوق.")
doc_col = common[0]

# عمود نص الكيان
text_col = next(
    (c for c in p.columns if c.lower() in ("text", "entity", "entity_text", "ent_text")),
    None,
)
# عمود النوع
label_col = next((c for c in p.columns if "label" in c.lower()), None)

if text_col is None or label_col is None:
    raise SystemExit("مش لاقي عمود النص أو النوع، ابعتي أسماء الأعمدة اللي فوق.")

p["entity_str"] = p[text_col].astype(str) + " (" + p[label_col].astype(str) + ")"
grouped = (
    p.groupby(doc_col)["entity_str"].apply(lambda s: ", ".join(s)).rename("entities")
)

out = d.merge(grouped, on=doc_col, how="left").head(20)
out.to_excel("results/emails_20_with_entities.xlsx", index=False)
print("تم! عدد الصفوف:", len(out))
print("الملف: results/emails_20_with_entities.xlsx")
