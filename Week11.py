import numpy as np
from matplotlib import pyplot as plt

all_avg_images = []

file_paths = ["full_numpy_bitmap_apple.npy", "full_numpy_bitmap_arm.npy","full_numpy_bitmap_axe.npy","full_numpy_bitmap_banana.npy",
"full_numpy_bitmap_baseball.npy","full_numpy_bitmap_bat.npy","full_numpy_bitmap_bicycle.npy","full_numpy_bitmap_bird.npy","full_numpy_bitmap_blueberry.npy",
"full_numpy_bitmap_book.npy"]

all_test_images = [] 

for file_path in file_paths:
    images = np.load(file_path).astype(np.float32)  

   
    indices = np.random.choice(len(images), size=10, replace=False)
    test_images = images[indices]
    train_images = np.delete(images, indices, axis=0)

   
    train_images_reshaped = np.reshape(train_images, (-1, 28, 28))

    avg_image = np.mean(train_images_reshaped, axis=0)
    all_avg_images.append(avg_image)
    all_test_images.append(test_images)
    

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(11,8))
for i in range(len(all_avg_images)):
    row, col = divmod(i, 5)
    axes[row, col].imshow(all_avg_images[i], cmap='gray') 
    axes[row, col].axis('off') 

# plt.tight_layout()
# plt.show()

test_index = 5
true_category = file_paths.index(file_paths[0])
test_image = all_test_images[true_category][test_index]  

scores = []
for avg_image in all_avg_images:
    dot_product = np.dot(test_image.flatten(),avg_image.flatten())
    scores.append(dot_product)

predict_category = np.argmax(scores)

print(scores)
print(f"Predicted category: {predict_category}")
print(f"True category: {true_category}")
print("predict is correct:"if predict_category == true_category else "predict is incorrect")

#predict category = 9
#true category = 0
#predict is incorrect 


#predict category = 0
#true category = 0
#predict is correct