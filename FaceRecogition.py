import os
import time
import cv2
import torch
import numpy as np
import pickle
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1, MTCNN
from torchvision import transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity

# 
BASE_DIR = "C:/Users/Ard Al Jood/Desktop/vs/facereco/people"
people = ["abdo","sabry","Naser","shrouk","jolie","maya","cr7","roaa"]

EMBED_FILE = "embeddings.pkl"
LABEL_FILE = "labels.pkl"
TH_FILE = "thresholds.pkl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# MTCNN for face detection (better cropping)
mtcnn = MTCNN(image_size=160, margin=10, device=device)

# Facenet model to get embeddings
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

to_tensor = transforms.ToTensor()

#  Uploding the images 
def load_image_paths(base_dir, people_list):
    image_paths = {}
    for person in people_list:
        folder = os.path.join(base_dir, person)
        if not os.path.isdir(folder):
            print(f"⚠ Folder not found for {person}: {folder}")
            image_paths[person] = []
            continue
        imgs = [os.path.join(folder, f) for f in os.listdir(folder)
                if f.lower().endswith(('.jpg','.jpeg','.png'))]
        image_paths[person] = imgs
    return image_paths

#  MTCCN Extract one face 
def extract_face(img_bgr):
    # img_bgr: frame from cv2 (BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # mtcnn returns PIL or tensor — use mtcnn to extract box and cropped face
    try:
        face_tensor = mtcnn(img_rgb)  # returns torch tensor (3,160,160) or None
    except Exception as e:
        print("MTCNN error:", e)
        face_tensor = None
    return face_tensor  # None or tensor on device (CPU/GPU)

# bulid embeddings from dataset folder
def build_embeddings_from_folder(base_dir, people_list, save=True):
    image_paths = load_image_paths(base_dir, people_list)
    embeddings = []
    labels = []
    per_person_embs = {}
    for person in tqdm(people_list, desc="Building embeddings"):
        per_person_embs[person] = []
        for img_path in image_paths[person]:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print("Can't read", img_path); continue
                face_tensor = extract_face(img)
                if face_tensor is None:
                    # If MTCNN didn't find face, try resize whole image (fallback)
                    small = cv2.resize(img, (160,160))
                    face_tensor = to_tensor(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
                else:
                    # mtcnn returns (3,160,160) tensor, ensure batch dim
                    if face_tensor.dim() == 3:
                        face_tensor = face_tensor.unsqueeze(0).to(device)
                    else:
                        face_tensor = face_tensor.to(device)

                with torch.no_grad():
                    emb = model(face_tensor).cpu().numpy().reshape(-1)  # (512,)
                embeddings.append(emb)
                labels.append(person)
                per_person_embs[person].append(emb)
            except Exception as e:
                print("Error processing", img_path, e)

        if len(per_person_embs[person]) == 0:
            print(f"⚠ No faces for {person}")
        else:
            per_person_embs[person] = np.vstack(per_person_embs[person])

    embeddings = np.vstack(embeddings) if len(embeddings)>0 else np.array([])
    labels = np.array(labels)
    # compute simple thresholds per person (mean cosine - margin)
    thresholds = {}
    margin = 0.12
    for p, arr in per_person_embs.items():
        if isinstance(arr, np.ndarray) and arr.shape[0] >= 2:
            sims = cosine_similarity(arr, arr)
            mask = ~np.eye(sims.shape[0], dtype=bool)
            sims_no_diag = sims[mask]
            thresholds[p] = float(np.mean(sims_no_diag) - margin)
        else:
            thresholds[p] = 0.55  # default
    # save
    if save:
        with open(EMBED_FILE, "wb") as f: pickle.dump(embeddings, f)
        with open(LABEL_FILE, "wb") as f: pickle.dump(labels, f)
        with open(TH_FILE, "wb") as f: pickle.dump(thresholds, f)
        print("Saved embeddings, labels and thresholds.")
    return embeddings, labels, thresholds

# Loading datasets
def load_data():
    if not (os.path.exists(EMBED_FILE) and os.path.exists(LABEL_FILE) and os.path.exists(TH_FILE)):
        return None, None, None
    with open(EMBED_FILE, "rb") as f: embeddings = pickle.load(f)
    with open(LABEL_FILE, "rb") as f: labels = pickle.load(f)
    with open(TH_FILE, "rb") as f: thresholds = pickle.load(f)
    return embeddings, labels, thresholds

#  Training KNN classifier 
def train_knn(embeddings, labels, n_neighbors=1):
    if embeddings is None or len(embeddings)==0:
        return None
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(embeddings, labels)
    return knn

#  Face prediction function
def predict_face(face_tensor, knn, embeddings, labels, thresholds):
    # face_tensor: torch tensor (1,3,160,160) on device
    with torch.no_grad():
        emb = model(face_tensor.to(device)).cpu().numpy().reshape(1,-1)  # (1,512)

    # if we have knn, use it
    if knn is not None:
        dists, idx = knn.kneighbors(emb, n_neighbors=1, return_distance=True)
        # knn with metric='cosine' returns distance in [0,2], where smaller = more similar
        best_idx = int(idx[0][0])
        pred_label = knn.predict(emb)[0]
        # compute cosine similarity manually to get score
        sim = cosine_similarity(emb, embeddings[best_idx].reshape(1,-1))[0][0] if embeddings is not None else 0.0
        # threshold check: per-person threshold else global
        th = thresholds.get(pred_label, 0.55)
        if sim >= th:
            return pred_label, float(sim)
        else:
            return "Unknown", float(sim)
    else:
        # fallback: compare with mean embeddings per person
        best_person = "Unknown"
        best_sim = -1.0
        for p, arr in per_person_mean.items():
            sim = float(cosine_similarity(emb, arr.reshape(1,-1))[0][0])
            if sim > best_sim:
                best_sim = sim
                best_person = p
        th = thresholds.get(best_person, 0.55)
        if best_sim >= th:
            return best_person, best_sim
        return "Unknown", best_sim

#  Live camera loop with enrollment support 
def run_camera(knn, embeddings, labels, thresholds):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera. Try changing index or check permissions.")
        return

    fps_time = time.time()
    frame_count = 0
    print("🎥 Camera started. Press 'q' to quit, 'n' to enroll detected face as new person, 's' to save DB.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Can't read frame")
            break
        frame_count += 1

        # detect face and landmarks/crop
        face_tensor = extract_face(frame)  # (3,160,160) or None
        name = "NoFace"
        score = 0.0

        if face_tensor is not None:
            if face_tensor.dim() == 3:
                face_tensor = face_tensor.unsqueeze(0).to(device)
            person, score = predict_face(face_tensor, knn, embeddings, labels, thresholds)
            # convert face_tensor back to numpy to draw bbox? We'll draw center label only
            # draw rectangle around whole frame center (optional) — better to get box from MTCNN but simplified here
            color = (0,255,0) if person != "Unknown" else (0,0,255)
            label_text = f"{person} ({score:.2f})"
            # show label
            cv2.putText(frame, label_text, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # FPS calc
        if frame_count % 10 == 0:
            now = time.time()
            fps = 10.0 / (now - fps_time + 1e-6)
            fps_time = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (20,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.imshow("Face Recognition (press q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # manual save DB
            with open(EMBED_FILE, "wb") as f: pickle.dump(embeddings, f)
            with open(LABEL_FILE, "wb") as f: pickle.dump(labels, f)
            with open(TH_FILE, "wb") as f: pickle.dump(thresholds, f)
            print("Database saved.")
        elif key == ord('n'):
            # enroll current face as new person
            if face_tensor is None:
                print("No face detected to enroll.")
                continue
            new_name = input("Enter new person's name (no spaces): ").strip()
            if new_name == "":
                print("Empty name. Cancel.")
                continue
            # compute embedding
            with torch.no_grad():
                new_emb = model(face_tensor.to(device)).cpu().numpy().reshape(-1)
            # save image & add to arrays
            person_folder = os.path.join(BASE_DIR, new_name)
            os.makedirs(person_folder, exist_ok=True)
            img_count = len([n for n in os.listdir(person_folder) if n.lower().endswith(('.jpg','.png'))]) if os.path.isdir(person_folder) else 0
            save_path = os.path.join(person_folder, f"{new_name}_{img_count+1}.jpg")
            # save cropped face image
            # convert face_tensor to image
            arr = (face_tensor.squeeze(0).cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, arr)
            # append to embeddings/labels
            if embeddings is None or len(embeddings)==0:
                embeddings = np.array([new_emb])
                labels = np.array([new_name])
            else:
                embeddings = np.vstack([embeddings, new_emb])
                labels = np.append(labels, new_name)
            # update thresholds for new_name (simple default)
            thresholds[new_name] = 0.55
            # retrain knn
            knn = train_knn(embeddings, labels, n_neighbors=1)
            print(f"Enrolled {new_name} and retrained KNN. Saved image to {save_path}")

    cap.release()
    cv2.destroyAllWindows()

#  main 
if __name__ == "__main__":
    # try load existing DB
    embeddings, labels, thresholds = load_data()
    if embeddings is None or labels is None:
        print("No DB found — building embeddings from folder. This may take a while.")
        embeddings, labels, thresholds = build_embeddings_from_folder(BASE_DIR, people, save=True)

    # train knn
    knn = train_knn(embeddings, labels, n_neighbors=1)
    # optional: compute per-person mean for fallback
    per_person_mean = {}
    if labels is not None and len(labels)>0:
        for p in np.unique(labels):
            per_person_mean[p] = np.mean(embeddings[labels==p], axis=0)
    # run camera
    run_camera(knn, embeddings, labels, thresholds)
