import face_recognition
import cv2
import os

# Path to known people's images
path = "C:/Users/Ard Al Jood/Desktop/vs/facereco/people"

known_faces = []
known_names = []



# Load and encode known faces

for root, dirs, files in os.walk(path):
    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(root, filename)

            # raed image
            image = face_recognition.load_image_file(img_path)

            # encoding
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_faces.append(encodings[0])

                # Extract person's name from directory structure
                person_name = os.path.basename(root)
                known_names.append(person_name)

print("photos are loaded:", len(known_faces), )

# Start the webcam
cap = cv2.VideoCapture(0)
print(" Camera started... Press 'Q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare with known faces
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_names[first_match_index]

        # Draw rectangle and name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    # Exit on 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  