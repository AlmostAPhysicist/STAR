#-----------------------
#-----------------------

#This code will load the video, extract frames, and convert them to grayscale for edge detection.

using VideoIO, Images

# Load the video
video_path = "Input Media/Run4.sloshing.h264"
cap = VideoIO.openvideo(video_path)

# Define frame storage and extract every Nth frame for efficiency
frames = []
N = 10  # Adjust this based on your video length and processing power

# Read frames from the video
frame_count = 0
while !eof(cap)
    frame = read(cap)
    if frame_count % N == 0  # Store every Nth frame
        push!(frames, Gray.(frame))  # Convert to grayscale for easier processing
    end
    frame_count += 1
end

close(cap)
println("Extracted $(length(frames)) frames.")


#-----------------------
#-----------------------

#For each frame, detect edges to identify the waterline. This example uses a Sobel edge detector from ImageEdgeDetection.jl.
## - Didn't work

# Apply edge detection to each frame. Apply the Sobel filter using imfilter from ImageFiltering.jl
using Images, ImageFiltering

# Obtain the predefined Sobel kernels
# sobel_x, sobel_y = Kernel.sobel()  # Returns the 3x3 Sobel kernels for x and y gradients
sobel_x = [-1 0 1; -2 0 2; -1 0 1]
sobel_y = [-1 -2 -1; 0 0 0; 1 2 1]
# Apply edge detection to each frame
edges = []
for frame in frames
    # Apply Sobel filter in both x and y directions
    gx = imfilter(frame, sobel_x)
    gy = imfilter(frame, sobel_y)

    # Combine the gradients to get edge intensity
    edge_frame = sqrt.(gx.^2 .+ gy.^2)
    push!(edges, edge_frame)
end

println("Edge detection completed.")

## Creating an edge detection video
# Define the video file path and parameters
# Define video parameters
# Initialize an empty vector to store converted frames
converted_frames = []

for frame in eachindex(edges) 
    # Initialize a matrix to store the converted pixels for the current frame
    cache = Matrix{Gray{N0f8}}(undef, size(edges[frame])) # Create a matrix with the same size

    for pixel in eachindex(edges[frame]) 
        # Clamp the value between 0.0 and 1.0
        clamped_value = clamp(edges[frame][pixel].val, 0.0f0, 1.0f0)
        # Convert the clamped value to Gray{N0f8} and store it in the cache
        cache[pixel] = Gray{N0f8}(clamped_value)
    end
    
    # Push the converted matrix for the current frame to the converted_frames
    push!(converted_frames, cache)
end

# Define video parameters
output_filename = "dummy_output_video.mp4"
framerate = 24
encoder_options = (crf=23, preset="medium")

# Open video output using the first converted frame to set the properties
open_video_out(output_filename, converted_frames[1], framerate=framerate, encoder_options=encoder_options) do writer
    for frame in converted_frames
        write(writer, frame)
    end
end

println("Video encoding complete: $output_filename")

#-----------------------
#-----------------------

# Once edges are detected, we can identify the waterline by finding prominent horizontal edges. To track the movement, we can use the ImageTracking.jl library.

using Images, ImageFiltering

# Calculate motion vectors using gradients between frames
motion_vectors = []

for i in 1:(length(edges)-1)
    # Compute frame difference as a proxy for motion vectors
    flow_x = edges[i+1] .- edges[i]
    flow_y = imfilter(flow_x, sobel_y)  # Vertical gradient
    flow_x = imfilter(flow_x, sobel_x)  # Horizontal gradient

    # Store motion vector (as a tuple of horizontal and vertical gradients)
    push!(motion_vectors, (flow_x, flow_y))
end

println("Motion vector calculation completed.")


#-----------------------
#-----------------------

# To compute wave characteristics, analyze the motion vectors for each frame. This involves calculating the average vector direction and magnitude.

# Analyze motion vectors for each part of the edge
amplitudes = []
directions = []

for flow in motion_vectors
    # Extract amplitude and direction from motion vectors
    amplitude = sqrt(sum(flow[1].^2 .+ flow[2].^2) / length(flow[1]))  # Average amplitude
    direction = atan(sum(flow[2]) / sum(flow[1]))  # Average direction
    push!(amplitudes, amplitude)
    push!(directions, direction)
end

println("Wave amplitude and direction analysis completed.")


#-----------------------
#-----------------------



using Plots

# Visualize motion vectors for a specific frame
frame_index = 1  # Choose the frame index to visualize

# Extract motion vectors for the chosen frame
flow_x = motion_vectors[frame_index][1]
flow_y = motion_vectors[frame_index][2]

# Define the grid for plotting vectors
rows, cols = size(flow_x)
x = repeat(1:cols, rows)'  # x-coordinates of the grid
y = repeat(1:rows, 1, cols)  # y-coordinates of the grid

# Scale motion vectors for better visualization
scale = 5.0
u = flow_x * scale  # Horizontal components of vectors
v = flow_y * scale  # Vertical components of vectors

# Plot the image and overlay the motion vectors
image_to_plot = edges[frame_index]  # Could be edges or original frame
heatmap(image_to_plot, color=:gray, title="Motion Vectors for Frame $frame_index")
quiver!(x, y, quiver=(u, v), color=:blue, arrow=:head)  # Overlay arrows

#-------------
#-------------


# If you want to train a model to recognize specific wave patterns, you can use Flux.jl to build a simple CNN or a regression model. This part would be more advanced and is optional based on your needs.

using Flux

# Define a simple model for wave pattern classification or regression
model = Chain(
    Conv((5, 5), 1 => 16, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 16 => 32, relu),
    MaxPool((2, 2)),
    flatten,
    Dense(32 * 8 * 8, 64),
    relu,
    Dense(64, 1)  # Output layer for regression (e.g., predicting amplitude)
)

# Dummy training setup for example
X_train = [edges[i] for i in 1:10]  # Substitute with your actual data
y_train = amplitudes[1:10]  # Substitute with your target amplitude data

loss(x, y) = Flux.mse(model(x), y)
opt = Flux.Descent(0.001)
Flux.train!(loss, params(model), zip(X_train, y_train), opt)

println("Model training completed.")
