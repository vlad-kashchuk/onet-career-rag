"""
ingest.py
Loads O*NET occupation data, converts it to text documents,
embeds them with sentence-transformers, and stores in ChromaDB.

Usage:
    python ingest.py
"""

from typing import Optional, List
import pandas as pd
from pathlib import Path
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path("data/raw")
CHROMA_DIR = "data/chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"

ONET_FILES = {
    "occupations":     "Occupation Data.txt",
    "skills":          "Skills.txt",
    "knowledge":       "Knowledge.txt",
    "abilities":       "Abilities.txt",
    "work_activities": "Work Activities.txt",
    "education":       "Education, Training, and Experience.txt",
}


def load_onet_file(filename: str) -> Optional[pd.DataFrame]:
    path = DATA_DIR / filename
    if not path.exists():
        print(f"  [skip] {filename} not found")
        return None
    return pd.read_csv(path, sep="\t", encoding="utf-8", low_memory=False)


def get_education_text(education_df: Optional[pd.DataFrame], onet_code: str) -> str:
    """Return the most-required education level for an occupation.

    O*NET stores education as multiple rows per occupation, each with a
    Data Value representing the % of workers who need that level.
    We filter for the 'Required Level of Education' element and return
    the category with the highest data value.
    """
    if education_df is None:
        return "N/A"

    edu_rows = education_df[education_df["O*NET-SOC Code"] == onet_code].copy()
    if edu_rows.empty:
        return "N/A"

    # Narrow to required-education rows if the column exists
    if "Element Name" in edu_rows.columns:
        req = edu_rows[edu_rows["Element Name"] == "Required Level of Education"]
        if not req.empty:
            edu_rows = req

    # Pick the category with the highest data value
    if "Data Value" in edu_rows.columns and "Category Description" in edu_rows.columns:
        valid = edu_rows.dropna(subset=["Data Value", "Category Description"]).copy()
        if not valid.empty:
            valid["Data Value"] = pd.to_numeric(valid["Data Value"], errors="coerce")
            valid = valid.dropna(subset=["Data Value"])
            if not valid.empty:
                top = valid.loc[valid["Data Value"].idxmax()]
                return str(top["Category Description"])

    # Fallback: first non-null description
    if "Category Description" in edu_rows.columns:
        first = edu_rows["Category Description"].dropna()
        if not first.empty:
            return str(first.iloc[0])

    return "N/A"


def build_occupation_docs() -> List[Document]:
    """Combine O*NET tables into one text document per occupation."""
    print("Loading O*NET files...")

    occupations = load_onet_file(ONET_FILES["occupations"])
    if occupations is None:
        raise FileNotFoundError(
            "Occupation Data.txt is required. "
            "Download the O*NET database from https://www.onetcenter.org/database.html "
            "and place the files in data/raw/"
        )

    skills_df      = load_onet_file(ONET_FILES["skills"])
    knowledge_df   = load_onet_file(ONET_FILES["knowledge"])
    abilities_df   = load_onet_file(ONET_FILES["abilities"])
    activities_df  = load_onet_file(ONET_FILES["work_activities"])
    education_df   = load_onet_file(ONET_FILES["education"])

    def top_items(df, onet_code, element_col="Element Name", value_col="Data Value", n=5):
        if df is None:
            return []
        subset = df[df["O*NET-SOC Code"] == onet_code].copy()
        if value_col in subset.columns:
            subset = subset.sort_values(value_col, ascending=False)
        return subset[element_col].dropna().unique().tolist()[:n]

    docs = []
    for _, row in occupations.iterrows():
        code  = row["O*NET-SOC Code"]
        title = row["Title"]
        desc  = row.get("Description", "")

        skills     = top_items(skills_df,    code)
        knowledge  = top_items(knowledge_df, code)
        abilities  = top_items(abilities_df, code)
        activities = top_items(activities_df, code)
        edu_text   = get_education_text(education_df, code)

        content = f"""Occupation: {title}
O*NET Code: {code}

Description:
{desc}

Top Skills: {", ".join(skills) if skills else "N/A"}

Knowledge Areas: {", ".join(knowledge) if knowledge else "N/A"}

Key Abilities: {", ".join(abilities) if abilities else "N/A"}

Work Activities: {", ".join(activities) if activities else "N/A"}

Education/Training: {edu_text}
"""

        docs.append(Document(
            page_content=content,
            metadata={"onet_code": code, "title": title}
        ))

    print(f"  Built {len(docs)} occupation documents")
    return docs


def ingest():
    print("=" * 50)
    print("Job/Career RAG — Data Ingestion")
    print("=" * 50)

    docs = build_occupation_docs()

    print(f"\nGenerating embeddings with '{EMBED_MODEL}'...")
    print("  (downloading model on first run, may take a minute...)")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    print("  Model loaded.")

    print(f"Storing in ChromaDB at '{CHROMA_DIR}'...")
    batch_size = 500
    vectorstore = None
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_DIR,
            )
        else:
            vectorstore.add_documents(batch)
        print(f"  Embedded {min(i + batch_size, len(docs))}/{len(docs)} documents")

    print("\nIngestion complete. Vector store is ready.")
    print(f"Total documents indexed: {len(docs)}")


if __name__ == "__main__":
    ingest()
