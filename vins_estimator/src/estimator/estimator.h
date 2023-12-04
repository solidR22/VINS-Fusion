/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once
 
#include <thread>
#include <mutex>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <ceres/ceres.h>
#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>

#include "parameters.h"
#include "feature_manager.h"
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../initial/solve_5pts.h"
#include "../initial/initial_sfm.h"
#include "../initial/initial_alignment.h"
#include "../initial/initial_ex_rotation.h"
#include "../factor/imu_factor.h"
#include "../factor/pose_local_parameterization.h"
#include "../factor/marginalization_factor.h"
#include "../factor/projectionTwoFrameOneCamFactor.h"
#include "../factor/projectionTwoFrameTwoCamFactor.h"
#include "../factor/projectionOneFrameTwoCamFactor.h"
#include "../featureTracker/feature_tracker.h"


class Estimator
{
  public:
    Estimator();
    ~Estimator();
    void setParameter();  // 设置从文件读取到的参数

    // interface
    void initFirstPose(Eigen::Vector3d p, Eigen::Matrix3d r);
    void inputIMU(double t, const Vector3d &linearAcceleration, const Vector3d &angularVelocity);                       // 对外接口：输入时间戳，加速度，角速度
    void inputFeature(double t, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &featureFrame);
    void inputImage(double t, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());                                   // 对外接口：输入时间戳，左图，右图
    void processIMU(double t, double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);
    void processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double header);
    void processMeasurements();                         // 设置参数的如果是多线程会时候打开
    void changeSensorType(int use_imu, int use_stereo);

    // internal
    void clearState();            // 构造函数的时候也会调用，初始化参数
    bool initialStructure();      // 单目+IMU初始化
    bool visualInitialAlign();    // 单目+IMU初始化
    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);  // 单目+IMU初始化
    void slideWindow();           // 滑动窗口法，数据移动，当数量达到10帧才会滑动
    void slideWindowNew();
    void slideWindowOld();
    void optimization();
    void vector2double();
    void double2vector();
    bool failureDetection(); // 跟踪失败
    bool getIMUInterval(double t0, double t1, vector<pair<double, Eigen::Vector3d>> &accVector, 
                                              vector<pair<double, Eigen::Vector3d>> &gyrVector);
    void getPoseInWorldFrame(Eigen::Matrix4d &T);
    void getPoseInWorldFrame(int index, Eigen::Matrix4d &T);
    void predictPtsInNextFrame();
    void outliersRejection(set<int> &removeIndex);
    double reprojectionError(Matrix3d &Ri, Vector3d &Pi, Matrix3d &rici, Vector3d &tici,
                                     Matrix3d &Rj, Vector3d &Pj, Matrix3d &ricj, Vector3d &ticj, 
                                     double depth, Vector3d &uvi, Vector3d &uvj);
    void updateLatestStates();
    // 使用上一时刻的姿态进行预积分
    void fastPredictIMU(double t, Eigen::Vector3d linear_acceleration, Eigen::Vector3d angular_velocity);
    bool IMUAvailable(double t);
    void initFirstIMUPose(vector<pair<double, Eigen::Vector3d>> &accVector); // 计算在第一帧图像时，IMU相对重力的旋转，并将航向角设置为0

    enum SolverFlag
    {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag
    {
        MARGIN_OLD = 0,           // 新帧为关键帧，marg掉最老帧
        MARGIN_SECOND_NEW = 1
    };

    std::mutex mProcess;
    std::mutex mBuf;
    std::mutex mPropagate;
    queue<pair<double, Eigen::Vector3d>> accBuf; // 加速度缓存
    queue<pair<double, Eigen::Vector3d>> gyrBuf; // 角速度缓存
    // 每一帧的特征，数据格式为:时间、特征点ID、相机ID(0/1)、归一化坐标、特征点坐标、归一化速度
    queue<pair<double, map<int, vector<pair<int, Eigen::Matrix<double, 7, 1> > > > > > featureBuf;
    double prevTime, curTime;                    // 前一帧和当前帧的时间
    bool openExEstimation;

    std::thread trackThread;
    std::thread processThread;

    // 跟踪类
    FeatureTracker featureTracker;

    SolverFlag solver_flag;  // 是否初始化的标志
    MarginalizationFlag  marginalization_flag;
    Vector3d g;

    Matrix3d ric[2];                         // 左右相机的Rbc
    Vector3d tic[2];                         // 左右相机的Tbc

    Vector3d        Ps[(WINDOW_SIZE + 1)];   // ? IMU Position（暂未预积分）Twb
    Vector3d        Vs[(WINDOW_SIZE + 1)];   // ? IMU 速度
    Matrix3d        Rs[(WINDOW_SIZE + 1)];   // ? IMU的旋转矩阵Rwb
    Vector3d        Bas[(WINDOW_SIZE + 1)];  // ? IMU acc bias
    Vector3d        Bgs[(WINDOW_SIZE + 1)];  // ? IMU gyr bias
    double td;                               // 时间偏移，设置在文件中，设为0                           

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    double Headers[(WINDOW_SIZE + 1)];

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)];        // 预积分
    Vector3d acc_0, gyr_0;  // 当前的加速度和角速度

    vector<double> dt_buf[(WINDOW_SIZE + 1)];                    // 每两个IMU数据的时间差
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)]; // IMU acc
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];    // IMU gyr

    int frame_count;
    int sum_of_outlier, sum_of_back, sum_of_front, sum_of_invalid;
    int inputImageCnt; // 输入的图片数量

    FeatureManager f_manager; // 在Estimator构造的时候就会构造f_manager
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu;  // 是否是第一次处理IMU数据
    bool is_valid, is_key;
    bool failure_occur;

    vector<Vector3d> point_cloud;
    vector<Vector3d> margin_cloud;
    vector<Vector3d> key_poses;
    double initial_timestamp;

    // Estimator::vector2double()
    double para_Pose[WINDOW_SIZE + 1][SIZE_POSE];                   // 位置和四元数
    double para_SpeedBias[WINDOW_SIZE + 1][SIZE_SPEEDBIAS];         // Vs, Bas, Bgs
    double para_Feature[NUM_OF_F][SIZE_FEATURE];                    // 特征点的深度值
    double para_Ex_Pose[2][SIZE_POSE];                              // 左右相机的Tbc
    double para_Retrive_Pose[SIZE_POSE];                            // *没用
    double para_Td[1][1];                                           // 同参数td，时间偏移量，设置为0
    double para_Tr[1][1];                                           // *没用

    int loop_window_index;

    MarginalizationInfo *last_marginalization_info;
    vector<double *> last_marginalization_parameter_blocks;         // 上一轮的残差数据

    map<double, ImageFrame> all_image_frame; // 所有的图像帧<时间戳，这一帧>
    IntegrationBase *tmp_pre_integration;    // 此刻预积分

    Eigen::Vector3d initP;
    Eigen::Matrix3d initR;

    // 最新的参数
    double latest_time;
    Eigen::Vector3d latest_P, latest_V, latest_Ba, latest_Bg, latest_acc_0, latest_gyr_0;
    Eigen::Quaterniond latest_Q;

    bool initFirstPoseFlag; // 第一帧位姿的初始化状态
    bool initThreadFlag;
};
