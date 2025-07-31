import os
import json
import shutil
from PIL import Image
import imagehash
import sys
from collections import defaultdict

class JewelryDuplicateDetector:
    def __init__(self, images_folder="HD", jsonl_file="final_saks.jsonl", duplicates_folder="duplicates"):
        self.images_folder = images_folder
        self.jsonl_file = jsonl_file
        self.duplicates_folder = duplicates_folder
        self.similarity_threshold = 8  # More lenient to catch more dups
        
        os.makedirs(duplicates_folder, exist_ok=True)
        
    def get_multi_hashes(self, image_path):
        """Extract multiple hashes for better duplicate detection"""
        try:
            img = Image.open(image_path)
            img.thumbnail((400, 400), Image.Resampling.LANCZOS)
            w, h = img.size
            
            # Multiple hash types for better accuracy
            hashes = {
                'phash': imagehash.phash(img, hash_size=8),
                'ahash': imagehash.average_hash(img, hash_size=8),
                'dhash': imagehash.dhash(img, hash_size=8),
                # Center crop for jewelry focus
                'center': imagehash.phash(img.crop((int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75))), hash_size=8)
            }
            return hashes
        except:
            return None
    
    def get_image_stats(self, image_path):
        """Pre-filtering stats"""
        try:
            size = os.path.getsize(image_path)
            # Looser size buckets to catch more potential duplicates
            size_bucket = size // 3000  # 3KB buckets instead of 5KB
            return size_bucket
        except:
            return None
    
    def calculate_similarity(self, hashes1, hashes2):
        """Multi-hash similarity with weighted scoring"""
        if not hashes1 or not hashes2:
            return 100  # Max difference
            
        # Calculate differences for each hash type
        diffs = {}
        for hash_type in hashes1:
            if hash_type in hashes2:
                diffs[hash_type] = hashes1[hash_type] - hashes2[hash_type]
        
        if not diffs:
            return 100
            
        # Weighted average (center hash is more important for jewelry)
        weights = {'phash': 1.0, 'ahash': 0.8, 'dhash': 0.6, 'center': 1.5}
        
        weighted_sum = 0
        total_weight = 0
        for hash_type, diff in diffs.items():
            weight = weights.get(hash_type, 1.0)
            weighted_sum += diff * weight
            total_weight += weight
            
        return weighted_sum / total_weight if total_weight > 0 else 100
    
    def find_duplicates(self):
        """Improved duplicate detection"""
        # Load dataset
        dataset = []
        with open(self.jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line.strip()))
        
        print(f"Loaded {len(dataset)} entries")
        
        # Pre-filter by file size (looser grouping)
        print("Pre-filtering by file size...")
        size_groups = defaultdict(list)
        
        for i, entry in enumerate(dataset):
            if 'target' not in entry:
                continue
            target_path = os.path.join(self.images_folder, entry['target'])
            if not os.path.exists(target_path):
                continue
                
            size_bucket = self.get_image_stats(target_path)
            if size_bucket is not None:
                # Include neighboring buckets to catch edge cases
                for bucket_offset in [-1, 0, 1]:
                    size_groups[size_bucket + bucket_offset].append((i, entry, target_path))
        
        # Get candidates from groups with multiple images
        candidates = []
        seen = set()
        for group in size_groups.values():
            if len(group) > 1:
                for item in group:
                    if item[0] not in seen:  # Avoid duplicates
                        candidates.append(item)
                        seen.add(item[0])
        
        print(f"Pre-filter: {len(candidates)} candidates from {len(dataset)} total")
        
        if len(candidates) == 0:
            print("No potential duplicates found")
            return [], {}, dataset
        
        # Extract hashes for candidates
        print("Extracting multi-hashes...")
        image_data = {}
        
        for idx, (i, entry, target_path) in enumerate(candidates):
            hashes = self.get_multi_hashes(target_path)
            if hashes:
                image_data[i] = {'hashes': hashes, 'entry': entry}
                
            if (idx + 1) % 25 == 0:
                print(f"Processed {idx + 1}/{len(candidates)} candidates", end='\r')
        
        print(f"\nSuccessfully processed {len(image_data)} images")
        
        # Find duplicates with improved algorithm
        print("Finding duplicate groups...")
        duplicate_groups = []
        processed = set()
        
        indices = list(image_data.keys())
        
        for i, idx1 in enumerate(indices):
            if idx1 in processed:
                continue
                
            group = [idx1]
            hashes1 = image_data[idx1]['hashes']
            
            for idx2 in indices[i+1:]:
                if idx2 in processed:
                    continue
                    
                hashes2 = image_data[idx2]['hashes']
                similarity = self.calculate_similarity(hashes1, hashes2)
                
                if similarity <= self.similarity_threshold:
                    group.append(idx2)
                    processed.add(idx2)
            
            if len(group) > 1:
                duplicate_groups.append(group)
                processed.update(group)
                print(f"Found duplicate group {len(duplicate_groups)}: {len(group)} images")
        
        print(f"\nTotal duplicate groups found: {len(duplicate_groups)}")
        return duplicate_groups, image_data, dataset
    
    def show_duplicates(self, duplicate_groups, image_data):
        """Simplified duplicate management"""
        if not duplicate_groups:
            print("✓ No duplicates found!")
            return []
            
        print(f"\n🎯 Found {len(duplicate_groups)} duplicate groups")
        
        # Show overview
        for i, group in enumerate(duplicate_groups):
            print(f"\nGroup {i+1}: {len(group)} similar images")
            for j, idx in enumerate(group):
                entry = image_data[idx]['entry']
                marker = "🏆 KEEP" if j == 0 else "🗑️ DELETE"
                print(f"  {marker} {entry.get('target', 'N/A')}")
        
        print(f"\n📋 OPTIONS:")
        print("1. 🚀 Delete all duplicates (keep first of each group)")
        print("2. 👁️ Review each group individually")
        print("3. ❌ Cancel")
        
        while True:
            choice = input("\nChoose option (1-3): ").strip()
            
            if choice == "1":
                return self.bulk_delete_all(duplicate_groups)
            elif choice == "2":
                return self.review_individually(duplicate_groups, image_data)
            elif choice == "3":
                print("❌ Cancelled")
                return []
            else:
                print("❌ Invalid choice")
    
    def bulk_delete_all(self, duplicate_groups):
        """Quick bulk delete"""
        total_to_delete = sum(len(group) - 1 for group in duplicate_groups)
        print(f"\n🚀 Will delete {total_to_delete} duplicates")
        
        confirm = input("Proceed? (y/n): ").lower()
        return duplicate_groups if confirm in ['yes', 'y'] else []
    
    def review_individually(self, duplicate_groups, image_data):
        """Review groups one by one"""
        confirmed = []
        
        for i, group in enumerate(duplicate_groups):
            print(f"\n📸 GROUP {i+1}/{len(duplicate_groups)}")
            
            for j, idx in enumerate(group):
                entry = image_data[idx]['entry']
                marker = "🏆 KEEP" if j == 0 else "🗑️ DELETE"
                print(f"{marker} {j+1}. {entry.get('target', 'N/A')}")
            
            while True:
                choice = input("Delete duplicates in this group? (y/n/v=view/q=quit): ").lower()
                
                if choice in ['y', 'yes']:
                    confirmed.append(group)
                    print("✅ Marked for deletion")
                    break
                elif choice in ['n', 'no']:
                    print("⏭️ Skipped")
                    break
                elif choice in ['v', 'view']:
                    self.view_images(group, image_data)
                elif choice in ['q', 'quit']:
                    return confirmed
                else:
                    print("❌ Invalid choice")
        
        return confirmed
    
    def view_images(self, group, image_data):
        """Simple image viewer"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, len(group), figsize=(4*len(group), 4))
            if len(group) == 1:
                axes = [axes]
            
            for i, idx in enumerate(group):
                entry = image_data[idx]['entry']
                img_path = os.path.join(self.images_folder, entry['target'])
                
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    axes[i].imshow(img)
                    axes[i].set_title(entry.get('target', 'Unknown'), fontsize=8)
                    axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Install matplotlib: pip install matplotlib")
        except Exception as e:
            print(f"Error: {e}")
    
    def process_duplicates(self, confirmed_groups, image_data, dataset):
        """Process confirmed duplicates"""
        if not confirmed_groups:
            print("No duplicates to process")
            return
            
        removed_indices = set()
        moved_count = 0
        
        print(f"Processing {len(confirmed_groups)} groups...")
        
        for group in confirmed_groups:
            # Keep first, move rest
            for idx in group[1:]:
                entry = image_data[idx]['entry']
                removed_indices.add(idx)
                
                # Move target file
                if 'target' in entry and entry['target']:
                    src = os.path.join(self.images_folder, entry['target'])
                    dst = os.path.join(self.duplicates_folder, entry['target'])
                    
                    if os.path.exists(src):
                        shutil.move(src, dst)
                        moved_count += 1
        
        # Create cleaned dataset
        cleaned_dataset = [entry for i, entry in enumerate(dataset) if i not in removed_indices]
        
        cleaned_file = self.jsonl_file.replace('.jsonl', '_cleaned.jsonl')
        with open(cleaned_file, 'w') as f:
            for entry in cleaned_dataset:
                f.write(json.dumps(entry) + '\n')
        
        print(f"✓ Moved {moved_count} duplicate files")
        print(f"✓ Cleaned dataset: {cleaned_file}")
        print(f"✓ Duplicates folder: {self.duplicates_folder}")
    
    def run(self):
        """Main execution"""
        print("🔍 Starting duplicate detection...")
        
        duplicate_groups, image_data, dataset = self.find_duplicates()
        
        if not duplicate_groups:
            print("✓ No duplicates found!")
            return
            
        confirmed = self.show_duplicates(duplicate_groups, image_data)
        self.process_duplicates(confirmed, image_data, dataset)

# Usage
if __name__ == "__main__":
    detector = JewelryDuplicateDetector()
    detector.run()