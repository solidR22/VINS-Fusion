/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);
// 根据误差去除追踪时的外点
void reduceVector(vector<cv::Point2f> &v, vector<uchar> status); 
void reduceVector(vector<int> &v, vector<uchar> status);
/**
 * 跟踪
*/
class FeatureTracker
{
public:
    FeatureTracker();
    // 返回每一帧的信息：特征点ID、相机ID、归一化坐标，特征点坐标，归一化速度
    map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat()); // 图像跟踪
    void setMask();
    void readIntrinsicParameter(const vector<string> &calib_file);
    void showUndistortion(const string &name);
    void rejectWithF();
    void undistortedPoints();
    // 计算归一化坐标，保存x和y
    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);
    void showTwoImage(const cv::Mat &img1, const cv::Mat &img2, 
                      vector<cv::Point2f> pts1, vector<cv::Point2f> pts2);
    void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                                   vector<int> &curLeftIds,
                                   vector<cv::Point2f> &curLeftPts, 
                                   vector<cv::Point2f> &curRightPts,
                                   map<int, cv::Point2f> &prevLeftPtsMap);
    void setPrediction(map<int, Eigen::Vector3d> &predictPts);
    double distance(cv::Point2f &pt1, cv::Point2f &pt2);
    void removeOutliers(set<int> &removePtsIds);
    cv::Mat getTrackImage();                     // 得到跟踪的图像
    bool inBorder(const cv::Point2f &pt);        // 将图像边缘的特征点去除

    int row, col;
    cv::Mat imTrack;
    cv::Mat mask;              // 用于限制提取特征点的距离
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img; // 上次和当前的图像
    vector<cv::Point2f> n_pts; // 这一帧新提取的特征点
    vector<cv::Point2f> predict_pts;
    vector<cv::Point2f> predict_pts_debug;
    vector<cv::Point2f> prev_pts, cur_pts, cur_right_pts; // 特征点：上一帧、这一帧、这一帧的右图
    vector<cv::Point2f> prev_un_pts, cur_un_pts, cur_un_right_pts;  // 归一化坐标的x,y
    vector<cv::Point2f> pts_velocity, right_pts_velocity;           // 存储这一帧的特征点的移动速度
    vector<int> ids, ids_right;                           // 特征点的ID
    vector<int> track_cnt;                                // 跟踪上的次数
    map<int, cv::Point2f> cur_un_pts_map, prev_un_pts_map;  // 当前帧和上一帧的归一化平面的坐标(x,y)
    map<int, cv::Point2f> cur_un_right_pts_map, prev_un_right_pts_map; // 同上，右图
    map<int, cv::Point2f> prevLeftPtsMap;                 // 上一帧左图的ID和特征点
    vector<camodocal::CameraPtr> m_camera;                // 相机模型
    double cur_time; // 当前时间
    double prev_time; // 上一次的时间
    bool stereo_cam;
    int n_id;
    bool hasPrediction; // 上一帧已经预测了下一帧的特征点（恒速模型）
};
