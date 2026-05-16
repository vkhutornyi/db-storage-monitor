import os
import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class NewFileHandler(FileSystemEventHandler):
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.last_processed = 0

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(('.xlsx', '.xls', '.csv')):
            filename = os.path.basename(event.src_path)
            print(f"\n🤖 [AI-Kernel] New file detected: {filename}")
            
            # Windows double-trigger protection filter
            current_time = time.time()
            if current_time - self.last_processed > 3:
                self.last_processed = current_time
                print("⏳ [AI-Kernel] Launching automatic forecast calculation...")
                time.sleep(1) # Allow file copy to complete
                
                try:
                    skills_dir = os.path.join(self.base_dir, 'skills')
                    result = subprocess.run(
                        ['python', 'data_processor.py'], 
                        cwd=skills_dir, 
                        capture_output=True, 
                        text=True, 
                        encoding='utf-8'
                    )
                    
                    if result.returncode == 0:
                        print(f"✅ [AI-Kernel] Execution Result: {result.stdout.strip()}")
                    else:
                        print(f"❌ [AI-Kernel] Skill Error Execution Failure:\n{result.stderr}")
                except Exception as e:
                    print(f"❌ [AI-Kernel] Unexpected Core Error: {e}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    WATCH_DIR = os.path.join(BASE_DIR, 'input_excel')
    
    print("="*60)
    print("   AI-AGENT KERNEL (OpenClaw style) STARTED SUCCESSFULLY")
    print(f" Monitoring folder: {WATCH_DIR}")
    print(" Agent status: ACTIVE. Drop a new file to update the dashboard.")
    print(" Press Ctrl + C to stop the agent.")
    print("="*60)

    event_handler = NewFileHandler(BASE_DIR)
    observer = Observer()
    observer.schedule(event_handler, path=WATCH_DIR, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n🛑 [AI-Kernel] Agent stopped by user.")
    observer.join()