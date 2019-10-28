import cv2, copy
import numpy as np
from imutils import paths


'''
# ORB Detector
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)
'''

query_path="../image/query/herslay1.jpeg"
db_path='../image/db'
query_img = cv2.imread(query_path)
match_score_list=[];db_img=None;
for path in list(paths.list_images(db_path)):    
    db_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    #'''
    # sift Detector
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(query_img, None)
    kp2, des2 = sift.detectAndCompute(db_img, None)
    #'
    # Brute Force Matching
    #bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    #bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)    
    #matching_result = cv2.drawMatches(query_img, kp1, db_img, kp2, matches, None, flags=2)
    match_score={'score':0, 'no_key_point':0,'kp1':None,'kp2':None,'matches':None,'img_path':'no_path'}
    for i,match in enumerate(matches):
        if match.distance <= 300:
            match_score['score']+=copy.copy(np.square(match.distance))
            match_score['no_key_point']=i+1;match_score['img_path']=path;    
            match_score['kp1']=copy.copy(kp1);match_score['kp2']=copy.copy(kp2); match_score['matches']=copy.copy(matches);
        else:
            break
    match_score['score']=np.sqrt(match_score['score'])
    temp=copy.copy(match_score)
    match_score_list.append(temp)

text='unmatched...';db_img_match_path='no_path';matching_result=None;match_score_key_point_min=10;
for match_score in match_score_list:
    
    if match_score['no_key_point'] > match_score_key_point_min:
        match_score_key_point_min=match_score['no_key_point']
        db_img_match_path=match_score['img_path']
        text='matched.. key point : '+str(match_score['no_key_point'])+' match_score :'+str(int(match_score['score']))
        kp1=match_score['kp1'];kp2=match_score['kp2']; matches=match_score['matches'];
        db_img = cv2.imread(db_img_match_path)
        #cv2.imshow("Img2", db_img) 
        matching_result = cv2.drawMatches(query_img, kp1, db_img, kp2, matches, None, flags=2)
        cv2.imshow("Matching result", matching_result)
        #cv2.waitKey(0)
    
cv2.putText(matching_result, text, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
cv2.imshow("query", query_img)
if db_img_match_path !='no_path':
    db_img = cv2.imread(db_img_match_path)
    cv2.imshow("db_img", db_img)    
    cv2.imshow("Matching result", matching_result)
    cv2.waitKey(0)
cv2.destroyAllWindows()
