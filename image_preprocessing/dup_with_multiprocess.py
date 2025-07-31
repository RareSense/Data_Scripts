import os
import json
import shutil
from PIL import Image
import imagehash
import sys
from collections import defaultdict
from multiprocessing import Pool
import functools

class JewelryDuplicateDetector:
    def __init__(self, images_folder="HD", jsonl_file="final_saks.jsonl", duplicates_folder="duplicates"):
        self.images_folder = images_folder
        self.jsonl_file = jsonl_file
        self.duplicates_folder = duplicates_folder
        self.similarity_threshold = 1.0
        
        os.makedirs(duplicates_folder, exist_ok=True)
        
    def get_tile_hashes(self, image_path):
        """Extract hash from center tile only (fastest + most accurate for jewelry)"""
        try:
            img = Image.open(image_path)
            img.thumbnail((300, 300), Image.Resampling.LANCZOS)  # Even smaller for speed
            w, h = img.size
            
            # Single center tile where jewelry typically appears
            center = img.crop((int(w*0.3), int(h*0.3), int(w*0.7), int(h*0.7)))
            
            return imagehash.phash(center, hash_size=8)  # Single hash
        except:
            return None
    
    def get_image_stats(self, image_path):
        """Get file size and basic histogram for pre-filtering"""
        try:
            size = os.path.getsize(image_path)
            
            # Quick color histogram for additional filtering
            img = Image.open(image_path)
            img.thumbnail((50, 50), Image.Resampling.LANCZOS)  # Very small for speed
            hist = img.histogram()
            
            # Simplified histogram signature (sum of RGB channels)
            r_sum = sum(hist[0:256])
            g_sum = sum(hist[256:512]) 
            b_sum = sum(hist[512:768])
            hist_signature = (r_sum, g_sum, b_sum)
            
            return size, hist_signature
        except:
            return None, None
    
    def process_batch(self, batch_data):
        """Process a batch of images for parallel processing"""
        results = {}
        for i, entry, target_path in batch_data:
            hash_val = self.get_tile_hashes(target_path)
            if hash_val:
                results[i] = {'hash': hash_val, 'entry': entry}
        return results
    
    def find_duplicates(self):
        """Optimized duplicate detection with multi-level filtering"""
        # Load dataset
        dataset = []
        with open(self.jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line.strip()))
        
        print(f"Loaded {len(dataset)} entries")
        
        # Step 1: Pre-filter by file size + color histogram
        print("Pre-filtering by file stats...")
        candidates_by_stats = defaultdict(list)
        valid_images = []
        
        for i, entry in enumerate(dataset):
            if 'target' not in entry:
                continue
            target_path = os.path.join(self.images_folder, entry['target'])
            if not os.path.exists(target_path):
                continue
                
            size, hist_sig = self.get_image_stats(target_path)
            if size and hist_sig:
                # Group by size bucket (5KB) + rough color similarity
                size_bucket = size // 5000
                color_bucket = (hist_sig[0] // 10000, hist_sig[1] // 10000, hist_sig[2] // 10000)
                key = (size_bucket, color_bucket)
                
                candidates_by_stats[key].append((i, entry, target_path))
                valid_images.append((i, entry, target_path))
        
        # Only keep groups with multiple images
        candidates = []
        for group in candidates_by_stats.values():
            if len(group) > 1:
                candidates.extend(group)
        
        print(f"Pre-filter: {len(candidates)} candidates from {len(valid_images)} valid images")
        
        if len(candidates) == 0:
            print("No potential duplicates found after pre-filtering")
            return [], {}, dataset
        
        # Step 2: Parallel hash extraction for candidates
        print("Extracting hashes (parallel processing)...")
        
        # Split candidates into batches for parallel processing
        batch_size = max(1, len(candidates) // 4)  # 4 processes
        batches = [candidates[i:i + batch_size] for i in range(0, len(candidates), batch_size)]
        
        image_data = {}
        try:
            with Pool(processes=4) as pool:
                results = pool.map(self.process_batch, batches)
                
            # Combine results
            for result in results:
                image_data.update(result)
                
        except Exception as e:
            print(f"Parallel processing failed, falling back to sequential: {e}")
            # Fallback to sequential processing
            for idx, (i, entry, target_path) in enumerate(candidates):
                hash_val = self.get_tile_hashes(target_path)
                if hash_val:
                    image_data[i] = {'hash': hash_val, 'entry': entry}
                    
                if (idx + 1) % 50 == 0:
                    print(f"Processed {idx + 1}/{len(candidates)} candidates", end='\r')
        
        print(f"\nSuccessfully processed {len(image_data)} images")
        
        # Step 3: Fast duplicate detection with single hash comparison
        print("Finding duplicate groups...")
        duplicate_groups = []
        processed = set()
        
        indices = list(image_data.keys())
        
        for i, idx1 in enumerate(indices):
            if idx1 in processed:
                continue
                
            group = [idx1]
            hash1 = image_data[idx1]['hash']
            
            for idx2 in indices[i+1:]:
                if idx2 in processed:
                    continue
                    
                hash2 = image_data[idx2]['hash']
                
                # Single hash comparison (much faster than multiple tiles)
                hash_diff = hash1 - hash2
                if hash_diff <= 5:  # Very similar hashes
                    group.append(idx2)
                    processed.add(idx2)
            
            if len(group) > 1:
                duplicate_groups.append(group)
                processed.update(group)
                print(f"Found duplicate group {len(duplicate_groups)}: {len(group)} images")
        
        print(f"\nTotal duplicate groups found: {len(duplicate_groups)}")
        return duplicate_groups, image_data, dataset
    
    def show_duplicates(self, duplicate_groups, image_data):
        """Enhanced user-friendly duplicate management"""
        if not duplicate_groups:
            print("✓ No duplicates found!")
            return []
            
        print(f"\n🎯 Found {len(duplicate_groups)} duplicate groups")
        print("=" * 60)
        
        # Show all groups first for overview
        for i, group in enumerate(duplicate_groups):
            print(f"Group {i+1}: {len(group)} similar images")
            for j, idx in enumerate(group):
                entry = image_data[idx]['entry']
                size = self.get_file_size_mb(os.path.join(self.images_folder, entry.get('target', '')))
                print(f"  {j+1}. {entry.get('target', 'N/A')} ({size}MB) - {entry.get('jewelry', 'N/A')}")
        
        print("\n" + "=" * 60)
        
        # Main menu
        while True:
            print(f"\n📋 DUPLICATE MANAGEMENT MENU")
            print("1. 🚀 Delete all duplicates (keep first of each group)")
            print("2. 👁️  Review each group individually") 
            print("3. 📁 View duplicates folder")
            print("4. 🔍 Preview specific group")
            print("5. ❌ Cancel (keep everything)")
            
            choice = input("\nChoose option (1-5): ").strip()
            
            if choice == "1":
                return self.bulk_delete_all(duplicate_groups)
            elif choice == "2":
                return self.review_individually(duplicate_groups, image_data)
            elif choice == "3":
                self.show_duplicates_folder()
            elif choice == "4":
                self.preview_specific_group(duplicate_groups, image_data)
            elif choice == "5":
                print("❌ Cancelled - no duplicates removed")
                return []
            else:
                print("❌ Invalid choice, please try again")
    
    def get_file_size_mb(self, filepath):
        """Get file size in MB"""
        try:
            size_bytes = os.path.getsize(filepath)
            return round(size_bytes / (1024 * 1024), 1)
        except:
            return 0.0
    
    def bulk_delete_all(self, duplicate_groups):
        """Simple bulk delete with basic confirmation"""
        total_to_delete = sum(len(group) - 1 for group in duplicate_groups)
        
        print(f"\n🚀 BULK DELETE MODE")
        print(f"Will keep first image of each group, delete {total_to_delete} duplicates")
        
        confirm = input("Proceed? (y/n): ").lower()
        if confirm in ['yes', 'y']:
            return duplicate_groups
        else:
            print("Cancelled")
            return []
    
    def review_individually(self, duplicate_groups, image_data):
        """Review each duplicate group one by one"""
        confirmed = []
        
        for i, group in enumerate(duplicate_groups):
            print(f"\n" + "="*50)
            print(f"📸 REVIEWING GROUP {i+1}/{len(duplicate_groups)}")
            print("="*50)
            
            # Show group details
            for j, idx in enumerate(group):
                entry = image_data[idx]['entry']
                size = self.get_file_size_mb(os.path.join(self.images_folder, entry.get('target', '')))
                keep_marker = "🏆 KEEP" if j == 0 else "🗑️  DELETE"
                print(f"{keep_marker} {j+1}. {entry.get('target', 'N/A')} ({size}MB)")
                print(f"     Prompt: {entry.get('prompt', 'N/A')[:70]}...")
                print(f"     Jewelry: {entry.get('jewelry', 'N/A')}")
                print()
            
            # Options for this group
            while True:
                print("Options:")
                print("✅ y = Confirm deletion (keep first, delete rest)")
                print("👁️  v = View images side by side")
                print("⏭️  s = Skip this group")
                print("🛑 q = Quit reviewing (save progress)")
                
                choice = input("Your choice (y/v/s/q): ").lower()
                
                if choice in ['y', 'yes']:
                    confirmed.append(group)
                    print(f"✅ Group {i+1} marked for deletion")
                    break
                elif choice in ['v', 'view']:
                    self.view_images_side_by_side(group, image_data)
                elif choice in ['s', 'skip']:
                    print(f"⏭️  Skipped group {i+1}")
                    break
                elif choice in ['q', 'quit']:
                    print(f"🛑 Stopped at group {i+1}")
                    return confirmed
                else:
                    print("❌ Invalid choice")
        
        print(f"\n✅ Review complete! {len(confirmed)} groups confirmed for deletion")
        return confirmed
    
    def preview_specific_group(self, duplicate_groups, image_data):
        """Preview a specific group by number"""
        while True:
            try:
                group_num = int(input(f"Enter group number (1-{len(duplicate_groups)}) or 0 to return: "))
                if group_num == 0:
                    return
                if 1 <= group_num <= len(duplicate_groups):
                    group = duplicate_groups[group_num - 1]
                    print(f"\n👀 Previewing Group {group_num}:")
                    for j, idx in enumerate(group):
                        entry = image_data[idx]['entry']
                        print(f"{j+1}. {entry.get('target', 'N/A')} - {entry.get('jewelry', 'N/A')}")
                    
                    if input("View images? (y/n): ").lower() in ['y', 'yes']:
                        self.view_images_side_by_side(group, image_data)
                    return
                else:
                    print(f"❌ Invalid group number. Enter 1-{len(duplicate_groups)}")
            except ValueError:
                print("❌ Please enter a valid number")
    
    def show_duplicates_folder(self):
        """Show contents of duplicates folder"""
        if not os.path.exists(self.duplicates_folder):
            print(f"📁 Duplicates folder doesn't exist yet: {self.duplicates_folder}")
            return
            
        files = os.listdir(self.duplicates_folder)
        if not files:
            print(f"📁 Duplicates folder is empty: {self.duplicates_folder}")
            return
            
        print(f"\n📁 DUPLICATES FOLDER: {self.duplicates_folder}")
        print(f"Contains {len(files)} files:")
        
        for i, file in enumerate(sorted(files)[:10], 1):  # Show first 10
            filepath = os.path.join(self.duplicates_folder, file)
            size = self.get_file_size_mb(filepath)
            print(f"{i:2d}. {file} ({size}MB)")
            
        if len(files) > 10:
            print(f"... and {len(files) - 10} more files")
            
        print(f"\n💾 Total folder size: {self.get_folder_size_mb(self.duplicates_folder)}MB")
        
        # Option to open folder
        if input("\n🔍 Open folder in file explorer? (y/n): ").lower() in ['y', 'yes']:
            try:
                import subprocess
                import platform
                
                if platform.system() == "Windows":
                    subprocess.run(["explorer", self.duplicates_folder])
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", self.duplicates_folder])
                else:  # Linux
                    subprocess.run(["xdg-open", self.duplicates_folder])
            except Exception as e:
                print(f"❌ Could not open folder: {e}")
                print(f"📁 Manual path: {os.path.abspath(self.duplicates_folder)}")
    
    def get_folder_size_mb(self, folder_path):
        """Get total folder size in MB"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(folder_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            return round(total_size / (1024 * 1024), 1)
        except:
            return 0.0
    
    def view_images_side_by_side(self, group, image_data):
        """Display images side by side using matplotlib"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, len(group), figsize=(5*len(group), 5))
            if len(group) == 1:
                axes = [axes]
            
            for i, idx in enumerate(group):
                entry = image_data[idx]['entry']
                img_path = os.path.join(self.images_folder, entry['target'])
                
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    axes[i].imshow(img)
                    axes[i].set_title(f"{entry.get('target', 'Unknown')}\n{entry.get('jewelry', 'N/A')}", fontsize=10)
                    axes[i].axis('off')
                else:
                    axes[i].text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
                    axes[i].set_title(entry.get('target', 'Unknown'))
                    axes[i].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Install matplotlib to view images: pip install matplotlib")
        except Exception as e:
            print(f"Error displaying images: {e}")
    
    def process_duplicates(self, confirmed_groups, image_data, dataset):
        """Clean duplicate processing"""
        if not confirmed_groups:
            print("No duplicates to process")
            return
            
        removed_indices = set()
        
        print(f"Processing {len(confirmed_groups)} duplicate groups...")
        
        for group in confirmed_groups:
            # Keep first image, move rest
            for idx in group[1:]:
                entry = image_data[idx]['entry']
                removed_indices.add(idx)
                
                # Move all related files
                for key in ['target', 'mask', 'ghost']:
                    if key in entry and entry[key]:
                        src = os.path.join(self.images_folder, entry[key])
                        dst = os.path.join(self.duplicates_folder, entry[key])
                        
                        if os.path.exists(src):
                            shutil.move(src, dst)
        
        # Create cleaned dataset
        cleaned_dataset = [entry for i, entry in enumerate(dataset) if i not in removed_indices]
        
        cleaned_file = self.jsonl_file.replace('.jsonl', '_cleaned.jsonl')
        with open(cleaned_file, 'w') as f:
            for entry in cleaned_dataset:
                f.write(json.dumps(entry) + '\n')
        
        print(f"✓ Complete! Removed {len(removed_indices)} duplicates")
        print(f"✓ Cleaned dataset: {cleaned_file}")
        print(f"✓ Moved files to: {self.duplicates_folder}")
    
    def run(self):
        """Main execution"""
        print("🔍 Starting optimized duplicate detection...")
        
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