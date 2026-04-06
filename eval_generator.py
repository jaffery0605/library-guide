import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

class EvalDatasetGenerator:
    def __init__(self, model_name="gemini-1.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model_name)
        self.output_file = "test_dataset.json"

    def generate_qna(self, file_path, num_questions=3):
        """
        Generates a set of Question-Answer pairs based on a document.
        This forms the 'Golden Dataset' for evaluation.
        """
        print(f"Generating {num_questions} Q&A pairs from {file_path}...")
        
        loader = TextLoader(file_path)
        docs = loader.load()
        content = docs[0].page_content

        system_prompt = (
            "You are an expert test engineer. Your task is to read the provided "
            "document content and generate high-quality Question and Answer pairs. "
            "The questions should be what a typical library user would ask. "
            "The answer MUST be strictly based on the provided content. "
            "\n\n"
            "Format the output as a JSON list of objects: "
            "[{\"question\": \"...\", \"ground_truth\": \"...\"}]"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Document Content:\n\n{content}")
        ])

        chain = prompt | self.llm
        
        try:
            response = chain.invoke({"content": content})
            # Extract JSON from the response (cleaning up any markdown formatting)
            raw_text = response.content.strip()
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            
            new_data = json.loads(raw_text)
            self._save_to_dataset(new_data)
            print(f"Successfully added {len(new_data)} pairs to the dataset.")
            
        except Exception as e:
            print(f"Error generating Q&A: {e}")

    def _save_to_dataset(self, new_data):
        existing_data = []
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r') as f:
                existing_data = json.load(f)
        
        existing_data.extend(new_data)
        
        with open(self.output_file, 'w') as f:
            json.dump(existing_data, f, indent=4)

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("Set GOOGLE_API_KEY in .env")
    else:
        gen = EvalDatasetGenerator()
        # Generate from our sample files
        gen.generate_qna("data/policies.md", num_questions=3)
        gen.generate_qna("data/database_guide.md", num_questions=2)
