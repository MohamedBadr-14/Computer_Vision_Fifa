import mss
import mss.tools
from datetime import datetime
import os

# Create a directory to save screenshots if it doesn't exist
output_dir = "screenshots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get the current timestamp for unique filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Initialize MSS
with mss.mss() as sct:
    # Check if monitor 2 exists
    if len(sct.monitors) > 2:  # Ensure monitor 2 exists (sct.monitors[0] is combined, 1 is first monitor, 2 is second)
        monitor = sct.monitors[2]  # Target monitor 2
        # Capture the screenshot for monitor 2
        screenshot = sct.grab(monitor)
        # Generate output filename
        output_file = os.path.join(output_dir, f"screenshot_monitor_2_{timestamp}.png")
        # Save the screenshot to a PNG file
        mss.tools.to_png(screenshot.rgb, screenshot.size, output=output_file)
        print(f"Saved screenshot for monitor 2 to {output_file}")
    else:
        print("Error: Monitor 2 not found. Please check your display configuration.")