import random
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import time
import matplotlib.pyplot as plt
import numpy as np

# Setup camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = HandDetector(maxHands=1)

# Game variables
timer = 0
stateResult = False
startGame = False
scores = [0, 0]  # [AI, Player]
roundsPlayed = 0
roundsToPlay = int(input("Enter number of rounds to play (minimum 15): "))
roundsToPlay = max(roundsToPlay, 15)

playerHistory = []  # Store player moves: 1=Rock, 2=Paper, 3=Scissors

# Function to predict next move using player history
def predict_move(history):
    if not history:
        return random.randint(1, 3)
    counts = [history.count(1), history.count(2), history.count(3)]
    total = sum(counts)
    probs = [c / total for c in counts]
    predicted_move = np.argmax(probs) + 1
    # Counter that move
    if predicted_move == 1:
        return 2  # Paper beats Rock
    elif predicted_move == 2:
        return 3  # Scissors beats Paper
    else:
        return 1  # Rock beats Scissors

while True:
    imgBG = cv2.imread("Resources/BG.png")
    success, img = cap.read()
    imgScaled = cv2.resize(img, (0, 0), None, 0.875, 0.875)
    imgScaled = imgScaled[:, 80:480]

    hands, img = detector.findHands(imgScaled)

    if startGame:
        if stateResult is False:
            timer = time.time() - initialTime
            cv2.putText(imgBG, str(int(timer)), (605, 435), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)

            if timer > 3:
                stateResult = True
                timer = 0

                playerMove = None
                if hands:
                    hand = hands[0]
                    fingers = detector.fingersUp(hand)
                    if fingers == [0, 0, 0, 0, 0]:
                        playerMove = 1  # Rock
                    elif fingers == [1, 1, 1, 1, 1]:
                        playerMove = 2  # Paper
                    elif fingers == [0, 1, 1, 0, 0]:
                        playerMove = 3  # Scissors

                if playerMove:
                    playerHistory.append(playerMove)

                    if roundsPlayed < 5:
                        aiMove = random.randint(1, 3)
                    else:
                        aiMove = predict_move(playerHistory)

                    imgAI = cv2.imread(f'Resources/{aiMove}.png', cv2.IMREAD_UNCHANGED)
                    imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

                    if (playerMove == 1 and aiMove == 3) or \
                        (playerMove == 2 and aiMove == 1) or \
                        (playerMove == 3 and aiMove == 2):
                        scores[1] += 1
                    elif (playerMove == 3 and aiMove == 1) or \
                            (playerMove == 1 and aiMove == 2) or \
                            (playerMove == 2 and aiMove == 3):
                        scores[0] += 1

                    roundsPlayed += 1
                    if roundsPlayed >= roundsToPlay:
                        break

    imgBG[234:654, 795:1195] = imgScaled

    if stateResult:
        imgBG = cvzone.overlayPNG(imgBG, imgAI, (149, 310))

    cv2.putText(imgBG, str(scores[0]), (410, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
    cv2.putText(imgBG, str(scores[1]), (1112, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)

    cv2.imshow("BG", imgBG)

    key = cv2.waitKey(1)
    if key == ord('s'):
        startGame = True
        initialTime = time.time()
        stateResult = False

# Release camera
cap.release()
cv2.destroyAllWindows()

# Display final results
print(f"Game Over. AI: {scores[0]} | Player: {scores[1]}")

# Plot probability density of player moves
plt.figure(figsize=(8, 5))
labels = ['Rock', 'Paper', 'Scissors']
counts = [playerHistory.count(1), playerHistory.count(2), playerHistory.count(3)]
plt.bar(labels, counts, color=['red', 'blue', 'green'])
plt.title('Player Move Distribution')
plt.xlabel('Move')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()