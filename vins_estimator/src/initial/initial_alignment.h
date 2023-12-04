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
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <ros/ros.h>
#include <map>
#include "../estimator/feature_manager.h"

using namespace Eigen;
using namespace std;

class ImageFrame
{
    public:
        ImageFrame(){};
        // 构造函数：这一帧的特征和时间戳
        ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& _points, double _t):t{_t},is_key_frame{false}
        {
            points = _points;
        };
        map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>> > > points;
        double t;
        Matrix3d R;                       // Rwb
        Vector3d T;                       // Twb
        IntegrationBase *pre_integration; // 预积分
        bool is_key_frame;
};
void solveGyroscopeBias(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs); // 计算gyr bias，对应公式10，11 mark
bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x);