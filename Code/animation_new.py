# %%
import pandas as pd
import cv2
import numpy as np

# %%
df = pd.read_csv('provided_data.csv')

# %%
df.head()

# %%
df.info()

# %% [markdown]
# ## Movement animation

# %%
def create_animation(df):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('animation.mp4', fourcc, 30.0, (800,600))
    
    # Normalize coordinates to fit within the frame
    x_min, x_max = df.iloc[:,1].min(), df.iloc[:,1].max()
    y_min, y_max = df.iloc[:,2].min(), df.iloc[:,2].max()

    for _, row in df.iterrows():
        frame = np.zeros((600,800,3), dtype=np.uint8)
        # Normalize and scale coordinates
        x = int((row.iloc[1] - x_min) / (x_max - x_min) * 780 + 10)
        y = int((row.iloc[2] - y_min) / (y_max - y_min) * 580 + 10)

        #draw the circle    
        cv2.circle(frame, (x,y), 5, (0,255,0), -1)

        # Add frame number text
        cv2.putText(frame, f"Frame: {int(row.iloc[0])}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)

    out.release()

# %%
create_animation(df)
print('Animation saved')

# %% [markdown]
# ## Speed and Acceleration

# %%
def speed_and_acceleration_animation(df):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('speed_acceleration_animation_short.mp4', fourcc, 30.0, (800,600))

    x_min, x_max = df.iloc[:,1].min(), df.iloc[:,1].max()
    y_min, y_max = df.iloc[:,2].min(), df.iloc[:,2].max()

    prev_x, prev_y = None, None
    prev_speed = None

    for i, row in df.iterrows():
        frame = np.zeros((600,800,3), dtype=np.uint8)
        
        # Normalize and scale coordinates
        x = int((row.iloc[1] - x_min) / (x_max - x_min) * 780 + 10)
        y = int((row.iloc[2] - y_min) / (y_max - y_min) * 580 + 10)
        
        if prev_x is not None and prev_y is not None:
            # Calculate speed (distance between current and previous points)
            distance = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
            speed = distance  # Assuming 1 unit time between frames
            
            # Calculate acceleration if we have previous speed
            if prev_speed is not None:
                acceleration = speed - prev_speed
            else:
                acceleration = 0
            
            # Use speed to change the color and size of the ball
            ball_size = max(5, int(speed * 2))  # Adjust scale for better visualization
            ball_color = (0, min(255, int(speed * 10)), max(0, 255 - int(speed * 10)))  # Green for fast, red for slow

            # Draw the ball
            cv2.circle(frame, (x, y), ball_size, ball_color, -1)
            
            # Show acceleration info (optional: draw arrows or text to indicate acceleration)
            cv2.putText(frame, f"Accel: {acceleration:.2f}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
        else:
            speed, acceleration = 0, 0  # No speed or acceleration for the first frame
        
        # Add frame number and speed text
        cv2.putText(frame, f"Frame: {int(row.iloc[0])}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Speed: {speed:.2f}", (10, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame to video
        out.write(frame)
        
        # Update previous values
        prev_x, prev_y = x, y
        prev_speed = speed

    out.release()


# %%
speed_and_acceleration_animation(df)
print('animation saved')

# %% [markdown]
# ## for sampling the dataset

# %%
short_df = df.iloc[:500,:]

# %%
short_df.shape

# %%
df.shape

# %%
speed_and_acceleration_animation(short_df)


