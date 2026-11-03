
import cv2
import os


NUM_IMAGES = 100


CONDITIONS = [
    "normal",
    "dark",
]



sign = input("Enter the ASL sign/letter to collect: ").strip().upper()

if sign == "":
    print("No sign entered. Program stopped.")
    exit()

print()
print("========================================")
print(f"Collecting ASL sign: {sign}")

print(f"Images per condition: {NUM_IMAGES}")
print(f"Total images: {NUM_IMAGES * len(CONDITIONS)}")
print("========================================")

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print()
print("Webcam opened successfully.")



for condition in CONDITIONS:

    
    folder = os.path.join(
        "dataset",
        sign,
        condition
    )

    os.makedirs(folder, exist_ok=True)

    count = 0

    print()
    print("----------------------------------------")
    print(f"Condition: {condition}")
    print(f"Folder: {folder}")
    print("----------------------------------------")

    input("Press ENTER to start this condition...")



    while count < NUM_IMAGES:

        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read webcam.")
            break

        

        progress_text = f"{condition.upper()}  |  {count}/{NUM_IMAGES}"

        cv2.putText(
            frame,
            progress_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            "SPACE = CAPTURE    Q = QUIT",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        
        cv2.imshow("ASL Dataset Collection", frame)

        

        key = cv2.waitKey(1) & 0xFF

        

        if key == ord(" "):

            count = count + 1

            filename = os.path.join(
                folder,
                f"{sign}_{condition}_{count:03d}.jpg"
            )

            # Save image
            cv2.imwrite(filename, frame)

            print(f"Saved: {filename}")
            print(f"Progress: {count}/{NUM_IMAGES}")

            
            captured_frame = frame.copy()

            cv2.putText(
                captured_frame,
                "CAPTURED!",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )

            cv2.imshow(
                "ASL Dataset Collection",
                captured_frame
            )
            cv2.waitKey(300)

    

        elif key == ord("q"):

            print()
            print("Dataset collection stopped by user.")

            camera.release()
            cv2.destroyAllWindows()

            exit()


    print()
    print("========================================")
    print(f"{condition.upper()} COMPLETED!")
    print(f"{count} images saved.")
    print("========================================")



camera.release()
cv2.destroyAllWindows()

print()
print("========================================")
print("DATASET COLLECTION COMPLETE!")
print("========================================")
print(f"Sign: {sign}")
print(f"Conditions: {len(CONDITIONS)}")
print(f"Images per condition: {NUM_IMAGES}")
print(f"Total images: {NUM_IMAGES * len(CONDITIONS)}")
print("========================================")