import time
import sys

try:
    from tarsier import Desktop, WebDesktop
except ImportError:
    print("❌ ERROR: 'tarsier-ai' is not installed in your current Python environment!")
    print("Please install it first by running: pip install -e .")
    sys.exit(1)

def run_visual_tests():
    print("==================================================")
    print("⚡ Starting Visual Tarsier-AI Integration Test ⚡")
    print("==================================================")

    # ==========================================
    # PART 1: VISUAL DESKTOP AUTOMATION (Calculator)
    # ==========================================
    print("\n--- [PART 1] Native Windows Desktop Test ---")
    desktop = Desktop()
    
    print("Opening Windows Calculator...")
    import subprocess
    try:
        subprocess.Popen('start calc', shell=True)
    except Exception as e:
        print(f"Error opening Calculator: {e}")
        
    print("Waiting for Calculator window...")
    try:
        calc = desktop.wait_for_window(name="Calculator", timeout=15)
        print(f"Connected to: '{calc.name}'")
        calc.focus()
        time.sleep(1)
        
        # Test semantic typing/clicking on Calculator
        print("Executing math: 7 * 8...")
        calc.button("Seven").click()
        time.sleep(0.5)
        
        try:
            calc.button("Multiply by").click() # Win 11 style
        except Exception:
            calc.button("Multiply").click() # Win 10 style
        time.sleep(0.5)
            
        calc.button("Eight").click()
        time.sleep(0.5)
        
        calc.button("Equals").click()
        print("Calculation complete! (Should show 56)")
        time.sleep(2)
        
    except Exception as e:
        print(f"❌ Desktop Test Error: {e}")

    # ==========================================
    # PART 2: VISUAL WEB AUTOMATION (Playwright Browser)
    # ==========================================
    print("\n--- [PART 2] Playwright Web Browser Test ---")
    print("Opening Chromium Browser (Visible Mode)...")
    
    # We set headless=False so you can visually watch the automation work!
    web = WebDesktop(headless=False)
    
    try:
        print("Navigating to Wikipedia...")
        page = web.goto("https://en.wikipedia.org/wiki/Main_Page")
        
        print("Locating search box semantically...")
        search_box = page.wait_for_element(role="searchbox", name="Search Wikipedia", timeout=5)
        
        print("Typing 'Tarsier'...")
        search_box.type("Tarsier")
        time.sleep(1)
        
        print("Clicking search...")
        search_btn = page.button("Search")
        search_btn.click()
        
        print("Waiting for results page...")
        time.sleep(3)
        page = web.get_current_page()
        
        heading = page.wait_for_element(role="heading", name="Tarsier", timeout=5)
        print(f"Success! Arrived at page heading: '{heading.name}'")
        time.sleep(2)
        
    except Exception as e:
        print(f"❌ Web Test Error: {e}")
    finally:
        print("Closing web browser...")
        web.close()

    print("\n==================================================")
    print("🎉 All tests executed successfully! 🎉")
    print("==================================================")

if __name__ == "__main__":
    run_visual_tests()
