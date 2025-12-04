#!/usr/bin/env python3
"""
Utility to select a pixel from an image or video and get its RGB/HSV values.
This helps determine color ranges for SelectColor.

Usage:
    python find_rgb.py <image_or_video_path>
"""

import cv2
import numpy as np
import sys
from pathlib import Path


class PixelColorPicker:
    def __init__(self, image_path: str):
        """
        Initialize the color picker.
        
        Args:
            image_path: Path to image or video file
        """
        self.image_path = Path(image_path)
        self.image = None
        self.selected_pixel = None
        self.window_name = "Click on a pixel to get RGB/HSV values (ESC to quit)"
        
        # Load image or first frame of video
        self._load_image()
    
    def _load_image(self):
        """Load image or first frame from video."""
        if not self.image_path.exists():
            raise FileNotFoundError(f"File not found: {self.image_path}")
        
        # Try to read as image first
        self.image = cv2.imread(str(self.image_path))
        
        # If not an image, try as video
        if self.image is None:
            cap = cv2.VideoCapture(str(self.image_path))
            if not cap.isOpened():
                raise IOError(f"Cannot open file: {self.image_path}")
            
            ret, frame = cap.read()
            if not ret:
                raise IOError(f"Cannot read first frame from: {self.image_path}")
            
            self.image = frame
            cap.release()
            print(f"Loaded first frame from video: {self.image_path}")
        else:
            print(f"Loaded image: {self.image_path}")
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback to handle clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get BGR values (OpenCV uses BGR, not RGB)
            bgr = self.image[y, x]
            b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
            
            # Convert to HSV
            pixel_bgr = np.uint8([[bgr]])
            hsv = cv2.cvtColor(pixel_bgr, cv2.COLOR_BGR2HSV)
            h, s, v = int(hsv[0, 0, 0]), int(hsv[0, 0, 1]), int(hsv[0, 0, 2])
            
            self.selected_pixel = (x, y, b, g, r, h, s, v)
            
            # Print values
            print("\n" + "="*60)
            print(f"Pixel at ({x}, {y}):")
            print(f"  BGR: ({b}, {g}, {r})")
            print(f"  RGB: ({r}, {g}, {b})")
            print(f"  HSV: ({h}, {s}, {v})")
            print("="*60)
            
            # Print suggested HSV range for SelectColor
            print("\nSuggested HSV range for SelectColor:")
            print(f"  Lower: np.array([{max(0, h-10)}, {max(0, s-50)}, {max(0, v-50)}])")
            print(f"  Upper: np.array([{min(179, h+10)}, 255, 255])")
            
            # Handle red wrap-around
            if h < 10 or h > 170:
                print("\n  Note: Red color wraps around in HSV!")
                print("  You may need two ranges:")
                if h < 10:
                    print(f"    Range 1: np.array([0, {max(0, s-50)}, {max(0, v-50)}]) to np.array([{h+10}, 255, 255])")
                    print(f"    Range 2: np.array([170, {max(0, s-50)}, {max(0, v-50)}]) to np.array([180, 255, 255])")
                else:
                    print(f"    Range 1: np.array([0, {max(0, s-50)}, {max(0, v-50)}]) to np.array([10, 255, 255])")
                    print(f"    Range 2: np.array([{h-10}, {max(0, s-50)}, {max(0, v-50)}]) to np.array([180, 255, 255])")
            
            print("\n" + "="*60)
    
    def run(self):
        """Run the interactive color picker."""
        if self.image is None:
            raise RuntimeError("No image loaded")
        
        # Create a copy for display
        display_image = self.image.copy()
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        print("\nInstructions:")
        print("  - Click on any pixel to get its RGB/HSV values")
        print("  - Press ESC or 'q' to quit")
        print("  - Press 's' to save the current selection")
        print()
        
        while True:
            # Draw crosshair at selected pixel if available
            if self.selected_pixel is not None:
                x, y = self.selected_pixel[0], self.selected_pixel[1]
                display_image = self.image.copy()
                
                # Draw crosshair
                cv2.line(display_image, (x-20, y), (x+20, y), (0, 255, 0), 2)
                cv2.line(display_image, (x, y-20), (x, y+20), (0, 255, 0), 2)
                cv2.circle(display_image, (x, y), 5, (0, 0, 255), -1)
                
                # Draw color patch
                b, g, r = self.selected_pixel[2], self.selected_pixel[3], self.selected_pixel[4]
                h, s, v = self.selected_pixel[5], self.selected_pixel[6], self.selected_pixel[7]
                
                # Color patch (BGR)
                patch_size = 50
                color_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                color_patch[:, :] = [b, g, r]
                
                # Place color patch in top-left corner
                h_img, w_img = display_image.shape[:2]
                if h_img > patch_size and w_img > patch_size:
                    display_image[10:10+patch_size, 10:10+patch_size] = color_patch
                    
                    # Add text
                    cv2.putText(display_image, f"RGB: ({r},{g},{b})", 
                               (10, 10+patch_size+20), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (255, 255, 255), 1)
                    cv2.putText(display_image, f"HSV: ({h},{s},{v})", 
                               (10, 10+patch_size+40), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.5, (255, 255, 255), 1)
            
            cv2.imshow(self.window_name, display_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or 'q'
                break
            elif key == ord('s') and self.selected_pixel is not None:
                # Save selection info to file
                output_file = self.image_path.parent / f"{self.image_path.stem}_color_info.txt"
                with open(output_file, 'w') as f:
                    x, y = self.selected_pixel[0], self.selected_pixel[1]
                    b, g, r = self.selected_pixel[2], self.selected_pixel[3], self.selected_pixel[4]
                    h, s, v = self.selected_pixel[5], self.selected_pixel[6], self.selected_pixel[7]
                    
                    f.write(f"Color Information from {self.image_path.name}\n")
                    f.write("="*60 + "\n")
                    f.write(f"Pixel location: ({x}, {y})\n")
                    f.write(f"BGR: ({b}, {g}, {r})\n")
                    f.write(f"RGB: ({r}, {g}, {b})\n")
                    f.write(f"HSV: ({h}, {s}, {v})\n")
                    f.write("\nSuggested HSV range for SelectColor:\n")
                    f.write(f"  Lower: np.array([{max(0, h-10)}, {max(0, s-50)}, {max(0, v-50)}])\n")
                    f.write(f"  Upper: np.array([{min(179, h+10)}, 255, 255])\n")
                
                print(f"\nColor information saved to: {output_file}")
        
        cv2.destroyAllWindows()
        
        if self.selected_pixel is not None:
            return self.selected_pixel
        return None


def main():
    """Main function."""
    
    image_path = "posEstimate/data/color_motion.mp4"
    
    try:
        picker = PixelColorPicker(image_path)
        result = picker.run()
        
        if result:
            print("\nFinal selection:")
            x, y = result[0], result[1]
            b, g, r = result[2], result[3], result[4]
            h, s, v = result[5], result[6], result[7]
            print(f"  Pixel: ({x}, {y})")
            print(f"  RGB: ({r}, {g}, {b})")
            print(f"  HSV: ({h}, {s}, {v})")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

