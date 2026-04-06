import json
import os

class HITLReviewer:
    def __init__(self, dataset_path="test_dataset.json"):
        self.dataset_path = dataset_path
        self.data = self._load_data()

    def _load_data(self):
        if not os.path.exists(self.dataset_path):
            print("No dataset found. Run eval_generator.py first.")
            return []
        with open(self.dataset_path, 'r') as f:
            return json.load(f)

    def _save_data(self):
        with open(self.dataset_path, 'w') as f:
            json.dump(self.data, f, indent=4)

    def run_review(self):
        """Iterate through unverified entries for human approval."""
        print(f"--- HITL Audit Session: {len(self.data)} items in dataset ---")
        
        for item in self.data:
            # Check if already verified
            if item.get("verified", False):
                continue

            print("\n" + "="*50)
            print(f"QUESTION: {item['question']}")
            print(f"AI ANSWER: {item['ground_truth']}")
            print("-" * 20)
            
            choice = input("[A]pprove | [E]dit | [R]eject | [S]kip | [Q]uit: ").lower()

            if choice == 'a':
                item['verified'] = True
                print("Item Approved!")
            elif choice == 'e':
                new_q = input(f"New Question (leave blank to keep): ") or item['question']
                new_a = input(f"New Answer (leave blank to keep): ") or item['ground_truth']
                item['question'] = new_q
                item['ground_truth'] = new_a
                item['verified'] = True
                print("Item Updated and Approved!")
            elif choice == 'r':
                self.data.remove(item)
                print("Item Rejected and Removed.")
            elif choice == 's':
                continue
            elif choice == 'q':
                break
            
            self._save_data() # Save after every action to prevent data loss

        print("\nReview session complete. Dataset updated.")

if __name__ == "__main__":
    reviewer = HITLReviewer()
    reviewer.run_review()
