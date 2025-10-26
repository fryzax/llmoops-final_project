import pandas as pd
import json
import matplotlib.pyplot as plt

# === 1️⃣ Charger ton CSV Langfuse ===
csv_path = "1761490957457-lf-traces-export-cmh270g6p000mad07qlm3ynvv.csv"
df = pd.read_csv(csv_path)

print("\n✅ CSV chargé avec succès.")
print(f"{len(df)} lignes détectées.")


# === 2️⃣ Parseur multi-niveaux pour la colonne 'output' ===
def parse_output(output_str):
    if not isinstance(output_str, str):
        return {}

    txt = output_str.strip()
    for i in range(4):  # on essaie plusieurs décodages
        try:
            txt = txt.replace('""', '"').replace('\\"', '"').strip('"')
            data = json.loads(txt)
            if isinstance(data, dict) and "metadata" in data:
                return data["metadata"]
            if isinstance(data, str):
                txt = data
                continue
        except Exception:
            continue
    return {}

# Application du parseur
df["parsed_metadata"] = df["output"].apply(parse_output)

# Vérif : affichage d’un exemple
print("\nExemple de metadata extraite :")
example = df["parsed_metadata"].iloc[0]
print(example if example else "⚠️ Vide — JSON non encore décodé.")


# === 3️⃣ Normaliser les metadata ===
meta_df = pd.json_normalize(df["parsed_metadata"])
meta_df.columns = [c.replace("metadata.", "") for c in meta_df.columns]
final_df = pd.concat([df, meta_df], axis=1)

print("\nColonnes détectées dans le DataFrame final :")
print(final_df.columns.tolist())

# === 4️⃣ Détection dynamique ===
num_cols = [c for c in final_df.columns if any(x in c for x in [
    "latency_seconds", "input_tokens", "output_tokens", "total_tokens", "article_word_count"
])]

if not num_cols:
    print("\n⚠️ Toujours aucune colonne numérique trouvée. Vérifions une ligne brute :")
    print(df["output"].iloc[0][:300])  # Affiche les 300 premiers caractères
else:
    print(f"\n✅ Colonnes numériques détectées : {num_cols}")
    for c in num_cols:
        final_df[c] = pd.to_numeric(final_df[c], errors="coerce")

    print("\n📊 Résumé statistique :")
    print(final_df[num_cols].describe())

    # === 5️⃣ Graphiques ===
    if "latency_seconds" in final_df.columns:
        plt.figure(figsize=(6,4))
        final_df["latency_seconds"].dropna().plot.hist(bins=10, edgecolor="black")
        plt.title("Distribution de la latence (secondes)")
        plt.xlabel("Latence (s)")
        plt.ylabel("Fréquence")
        plt.tight_layout()
        plt.show()

    if all(x in final_df.columns for x in ["article_word_count", "latency_seconds"]):
        plt.figure(figsize=(6,4))
        plt.scatter(final_df["article_word_count"], final_df["latency_seconds"])
        plt.title("Latence vs Longueur d'article")
        plt.xlabel("Nombre de mots")
        plt.ylabel("Latence (s)")
        plt.tight_layout()
        plt.show()

# === 6️⃣ Sauvegarde du dataset nettoyé ===
final_df.to_csv("clean_langfuse_data.csv", index=False)
print("\n💾 Données nettoyées enregistrées sous : clean_langfuse_data.csv")
