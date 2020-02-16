import cv2
import os 

for image_file in os.listdir("test"):
    full_file_path = os.path.join("test", image_file)
    img = cv2.imread(full_file_path, cv2.IMREAD_UNCHANGED)
 
    print('Original Dimensions : ',img.shape)
    
    scale_percent = 20 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    print('Resized Dimensions : ',resized.shape)
    
    cv2.imshow("Resized image", resized)

    full_write_path = os.path.join("low_test", image_file)
    cv2.imwrite(full_write_path,resized)


cv2.waitKey(0)
cv2.destroyAllWindows()