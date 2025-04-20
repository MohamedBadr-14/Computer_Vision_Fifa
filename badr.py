import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from ast import literal_eval

def draw_scene_analysis(image_path, scene_data):
    """
    Draw bounding boxes and transformed positions on a saved frame.
    
    Args:
        image_path: Path to the image file
        scene_data: Dictionary containing scene analysis results
    
    Returns:
        Annotated image
    """
    # Load the image
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Create a copy to draw on
    annotated_frame = frame.copy()
    
    # Define colors for different object types (BGR format)
    colors = {
        'players': {
            1: (0, 255, 0),    # Team 1: Green
            2: (0, 0, 255)     # Team 2: Red
        },
        'ball': (0, 165, 255),  # Orange
        'goalkeeper': (255, 0, 255),  # Magenta
        'goalpost': (255, 255, 0)     # Cyan
    }
    
    # Field dimensions for visualization of transformed positions
    field_width = 80
    field_height = 120
    field_scale = 8  # pixels per unit
    field_img = np.ones((field_height * field_scale, field_width * field_scale, 3), dtype=np.uint8) * 50
    cv2.rectangle(field_img, (0, 0), (field_width * field_scale, field_height * field_scale), (0, 100, 0), -1)
    
    # Draw midfield line
    cv2.line(field_img, (0, field_height * field_scale // 2), 
             (field_width * field_scale, field_height * field_scale // 2), 
             (255, 255, 255), 2)
    
    # Draw center circle
    cv2.circle(field_img, (field_width * field_scale // 2, field_height * field_scale // 2), 
               10 * field_scale // 2, (255, 255, 255), 2)
    
    # Draw goal areas
    cv2.rectangle(field_img, 
                 (field_width * field_scale // 2 - 8 * field_scale, 0), 
                 (field_width * field_scale // 2 + 8 * field_scale, 6 * field_scale), 
                 (255, 255, 255), 2)
    cv2.rectangle(field_img, 
                 (field_width * field_scale // 2 - 8 * field_scale, field_height * field_scale - 6 * field_scale), 
                 (field_width * field_scale // 2 + 8 * field_scale, field_height * field_scale), 
                 (255, 255, 255), 2)
    
    # Track transformed positions for the field view
    transformed_positions = []
    
    # Draw player bounding boxes
    if 'players' in scene_data:
        for player_id, player_data in scene_data['players'].items():
            team = player_data.get('team', 0)
            color = colors['players'].get(team, (200, 200, 200))  # Default gray for unknown team
            
            # Draw bounding box
            bbox = player_data['bbox']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add player ID and team information
            has_ball = "🏀" if player_data.get('has_ball', False) else ""
            label = f"P{player_id} (T{team}) {has_ball}"
            cv2.putText(annotated_frame, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Collect transformed position for field view
            if 'position_transformed' in player_data:
                x_t, y_t = player_data['position_transformed']
                transformed_positions.append({
                    'type': 'player',
                    'team': team,
                    'id': player_id,
                    'has_ball': player_data.get('has_ball', False),
                    'pos': (x_t, y_t),
                    'color': color
                })
    
    # Draw ball bounding box
    if 'ball' in scene_data:
        for ball_id, ball_data in scene_data['ball'].items():
            color = colors['ball']
            
            # Draw bounding box if it exists
            if 'bbox' in ball_data:
                bbox = ball_data['bbox']
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add ball label
                label = f"Ball ({ball_data.get('conf', 0):.2f})"
                cv2.putText(annotated_frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Collect transformed position
            if 'position_transformed' in ball_data:
                x_t, y_t = ball_data['position_transformed']
                transformed_positions.append({
                    'type': 'ball',
                    'id': ball_id,
                    'pos': (x_t, y_t),
                    'color': color
                })
    
    # Draw goalkeeper bounding box
    if 'goalkeeper' in scene_data:
        for gk_id, gk_data in scene_data['goalkeeper'].items():
            color = colors['goalkeeper']
            
            # Draw bounding box
            if 'bbox' in gk_data:
                bbox = gk_data['bbox']
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add goalkeeper label
                label = f"GK {gk_id}"
                cv2.putText(annotated_frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Collect transformed position
            if 'position_transformed' in gk_data:
                x_t, y_t = gk_data['position_transformed']
                transformed_positions.append({
                    'type': 'goalkeeper',
                    'id': gk_id,
                    'pos': (x_t, y_t),
                    'color': color
                })
    
    # Draw goalpost bounding box
    if 'goalpost' in scene_data:
        for gp_id, gp_data in scene_data['goalpost'].items():
            color = colors['goalpost']
            
            # Draw bounding box
            if 'bbox' in gp_data:
                bbox = gp_data['bbox']
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add goalpost label
                label = f"Goal ({gp_data.get('conf', 0):.2f})"
                cv2.putText(annotated_frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Collect transformed position
            if 'position_transformed' in gp_data:
                x_t, y_t = gp_data['position_transformed']
                transformed_positions.append({
                    'type': 'goalpost',
                    'id': gp_id,
                    'pos': (x_t, y_t),
                    'color': color
                })
    
    # Draw transformed positions on field visualization
    for obj in transformed_positions:
        x_t, y_t = obj['pos']
        # Convert to pixel coordinates in our visualization
        x_pixel = int(x_t * field_scale)
        y_pixel = int(y_t * field_scale)
        
        # Draw circle for the object
        radius = 8 if obj['type'] == 'ball' else 12
        cv2.circle(field_img, (x_pixel, y_pixel), radius, obj['color'], -1)
        
        # Add label
        if obj['type'] == 'player':
            label = f"P{obj['id']}"
            if obj.get('has_ball', False):
                label += "🏀"
        elif obj['type'] == 'goalkeeper':
            label = f"GK{obj['id']}"
        elif obj['type'] == 'goalpost':
            label = f"G{obj['id']}"
        else:
            label = f"B{obj['id']}"
            
        cv2.putText(field_img, label, (x_pixel + 5, y_pixel + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(field_img, label, (x_pixel + 5, y_pixel + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, obj['color'], 1)
    
    # Add frame number and timestamp to the image
    frame_name = os.path.basename(image_path)
    cv2.putText(annotated_frame, f"Frame: {frame_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Resize the field visualization to fit below the frame
    field_height_target = int(annotated_frame.shape[0] * 0.4)  # 40% of the frame height
    field_width_target = annotated_frame.shape[1]
    field_img_resized = cv2.resize(field_img, (field_width_target, field_height_target))
    
    # Combine original frame with field visualization
    combined_img = np.vstack([annotated_frame, field_img_resized])
    
    return combined_img

# Example scene data from the log (you provided this)
scene_data_str = """{'players': {81: {'bbox': [155.5291748046875, 1024.72900390625, 231.714599609375, 1193.01220703125], 'conf': 0.69391733, 'position': (193, 1193), 'position_transformed': [29.88883399963379, 56.17292022705078], 'team': 1}, 86: {'bbox': [2458.92529296875, 642.0350341796875, 2532.72607421875, 760.2269287109375], 'conf': 0.18044665, 'position': (2495, 760), 'position_transformed': [56.1196403503418, 48.03258514404297], 'team': 2}}, 'referees': {}, 'ball': {1: {'bbox': [351.0340270996094, 1175.923583984375, 390.5963439941406, 1212.489501953125], 'conf': 0.11, 'position': (370, 1194), 'position_transformed': [31.418243408203125, 56.48139572143555]}}, 'goalkeeper': {}, 'goalpost': {}}"""

# Parse the string into a Python dictionary
scene_data = literal_eval(scene_data_str)

# Usage with a sample image path - replace with your actual frame path
# This should be one of the frames from your captured_frames_20250420_134441 folder
image_path = "D:/fifa-vision/captured_frames_20250420_143316/frame_buffer_000176.jpg"

# Generate the visualization
try:
    annotated_image = draw_scene_analysis(image_path, scene_data)
    
    # Save the result
    output_path = "D:/fifa-vision/annotated_frame.jpg"
    cv2.imwrite(output_path, annotated_image)
    print(f"Annotated image saved to {output_path}")
    
    # Display the result (optional - comment out if running in a non-interactive environment)
    plt.figure(figsize=(15, 12))
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Error processing image: {e}")