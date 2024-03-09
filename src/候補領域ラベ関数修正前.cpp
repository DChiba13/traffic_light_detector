#include <opencv2/opencv.hpp>
#include <experimental/filesystem>
#include <ryusei/common/logger.hpp>
#include <ryusei/common/defs.hpp>
#include <fstream>

using namespace project_ryusei;
using namespace cv;
using namespace std;
namespace pr = project_ryusei;
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
namespace fs = std::experimental::filesystem;

// 赤青黃のHSV表色系での閾値を設定
constexpr int MIN_H_RED_01 = 1;
constexpr int MAX_H_RED_01 = 10;
constexpr int MIN_H_RED_02 = 165;
constexpr int MAX_H_RED_02 = 180;
constexpr int MIN_S_RED = 35;
constexpr int MAX_S_RED = 255;
constexpr int MIN_V_RED = 40;
constexpr int MAX_V_RED = 255;

constexpr int MIN_H_GREEN = 60;
constexpr int MAX_H_GREEN = 95;
constexpr int MIN_S_GREEN = 35;
constexpr int MAX_S_GREEN = 255;
constexpr int MIN_V_GREEN = 40;
constexpr int MAX_V_GREEN = 255;

constexpr int MIN_H_YELLOW = 0;
constexpr int MAX_H_YELLOW = 60;
constexpr int MIN_S_YELLOW = 40;
constexpr int MAX_S_YELLOW = 255;
constexpr int MIN_V_YELLOW = 145;
constexpr int MAX_V_YELLOW = 255;

// 信号が青なのか赤なのか判断するフラグ
bool green_light_flag = false;
bool red_light_flag = false;

// 画像の上から何％の高さを表示するのか設定　
constexpr float IMAGE_ABOVE_RASIO = .33f;

// IMAGE_THRESHフレーム連続で赤、青が認識されると信号とみなす
constexpr int RED_IMAGE_THRESH = 0;
constexpr int GREEN_IMAGE_THRESH = 0;

// 赤、青信号が何フレーム連続で検出されたか数えるcount
int green_cnt = 0;
int red_cnt = 0;

// 赤or青判定を画像に表示する文字
std::string light_msg_state;

// 信号の候補領域のピクセル数の閾値
int pixel_num = 0;
constexpr int MIN_PIX_NUM = 200;
constexpr int MAX_PIX_NUM = 1000;

// 信号の候補領域のアスペクト比の閾値
// 横 : 縦 = ASPECT_RATIO : 1
double aspect_ratio = .0f;
constexpr double MIN_ASPECT_RATIO = 0.8;
constexpr double MAX_ASPECT_RATIO = 1.4;

/* ファイル名を取得 */
void getFiles(const fs::path &path, const string &extension, vector<fs::path> &files)
{
  for(const fs::directory_entry &p : fs::directory_iterator(path)){
    if(!fs::is_directory(p)){
      if(p.path().extension().string() == extension){
        files.push_back(p);
      }
    }
  }
  sort(files.begin(), files.end());
}

/* カメラ画像から赤色の画素を抽出する関数 */
void extractRedSignal(cv::Mat &rgb, cv::Mat &hsv, cv::Mat &extract_red)
{
  // 赤信号の赤色を抽出
  for(int y = 0; y < rgb.rows; y++){
      for(int x = 0; x < rgb.cols; x++){
        cv::Vec3b val = hsv.at<cv::Vec3b>(y,x);
        if(  /*(MIN_H_RED_01 <= val[0] && val[0] <= MAX_H_RED_01) || */
            (MIN_H_RED_02 <= val[0] && val[0] <= MAX_H_RED_02)
          && MIN_S_RED <= val[1] && val[1] <= MAX_S_RED
          && MIN_V_RED <= val[2] && val[2] <= MAX_V_RED)
        {
          extract_red.at<cv::Vec3b>(y,x) = rgb.at<cv::Vec3b>(y,x);
        }
      }
  }
  // cv::imshow("extract_red",extract_red);
}

/* カメラ画像から緑色の画素を抽出する関数 */
void extractGreenSignal(cv::Mat &rgb, cv::Mat &hsv, cv::Mat &extract_green)
{
  // 青信号の緑色を抽出
  for(int y = 0; y < rgb.rows; y++){
      for(int x = 0; x < rgb.cols; x++){
        cv::Vec3b val = hsv.at<cv::Vec3b>(y,x);
        if(    MIN_H_GREEN <= val[0] && val[0] <= MAX_H_GREEN
            && MIN_S_GREEN <= val[1] && val[1] <= MAX_S_GREEN
            && MIN_V_GREEN <= val[2] && val[2] <= MAX_V_GREEN)
          {
            extract_green.at<cv::Vec3b>(y, x) = rgb.at<cv::Vec3b>(y, x);
        }
      }
  }
  // cv::imshow("extract_green",extract_green);
}

/* 抽出した色を白くし、二値化する関数 */
void binalizeImage(cv::Mat &src, cv::Mat &gray_img)
{
  for(int y = 0; y<src.rows; y++)
  {
    for(int x = 0; x<src.cols; x++)
    {
      if(src.at<cv::Vec3b>(y,x)!=cv::Vec3b(0, 0, 0))
      {
        gray_img.at<uchar>(y,x) = 255;
      }
    }
  }
}

/* ピンク色または水色の中に黄色が見えたら赤or青色の矩形でラベリングする関数 */
/* isRedLightがtrueなら赤信号用の処理、falseなら青信号用の処理になる */
void extractYellowInBlob(cv::Mat &rgb, cv::Mat &bin_img, int num_labels, const std::vector<int> &widths, const std::vector<int> &heights, const std::vector<int> &lefts, const std::vector<int> &tops, bool isRedSignal, cv::Mat top_region)
{
  cv::Mat hsv;
  cv::cvtColor(rgb, hsv, cv::COLOR_BGR2HSV);
  for (int label = 0; label < num_labels; label++)
  {
    int width = widths[label];
    int height = heights[label];
    int left = lefts[label];
    int top = tops[label];

    // ピクセル数とアスペクト比を見る
    pixel_num = height * width;
    aspect_ratio = ((double)width)/((double)height);
    if(pixel_num<MIN_PIX_NUM || pixel_num>MAX_PIX_NUM)
    {
      continue;
    }
    if(aspect_ratio<MIN_ASPECT_RATIO || aspect_ratio>MAX_ASPECT_RATIO)
    {
      continue;
    }

    cv::Mat blob_rgb(rgb,cv::Rect(left, top, width, height));
    // cv::imshow("rgb",blob_rgb);
    cv::Mat blob_hsv(hsv, cv::Rect(left, top, width, height));

    cv::Mat extract_yellow = cv::Mat::zeros(blob_hsv.size(), blob_hsv.type());
    cv::medianBlur(blob_hsv, blob_hsv, 3);
    for (int y = 0; y < blob_hsv.rows; y++)
    {
      for (int x = 0; x < blob_hsv.rows; x++)
      {
        cv::Vec3b val = blob_hsv.at<cv::Vec3b>(y, x);
        if (   MIN_H_YELLOW <= val[0] && val[0] <= MAX_H_YELLOW
            && MIN_S_YELLOW <= val[1] && val[1] <= MAX_S_YELLOW
            && MIN_V_YELLOW <= val[2] && val[2] <= MAX_V_YELLOW)
        {
          extract_yellow.at<cv::Vec3b>(y, x) = blob_rgb.at<cv::Vec3b>(y, x);
        }
      }
    }
    // cv::imshow("yellow",extract_yellow);

    cv::Mat bin_img_yellow = cv::Mat::zeros(blob_hsv.size(), CV_8UC1);
    binalizeImage(extract_yellow, bin_img_yellow);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    cv::Mat labeled_yellow;
    cv::Mat stats_yellow, centroids_yellow;
    int num_labels_yellow = cv::connectedComponentsWithStats(bin_img_yellow, labeled_yellow, stats_yellow, centroids_yellow);
    for (int label = 0; label < num_labels_yellow; label++) 
    {
      int yellow_width = stats_yellow.at<int>(label, cv::CC_STAT_WIDTH);
      int yellow_height = stats_yellow.at<int>(label, cv::CC_STAT_HEIGHT);
      int yellow_left = stats_yellow.at<int>(label, cv::CC_STAT_LEFT);
      int yellow_top = stats_yellow.at<int>(label, cv::CC_STAT_TOP);

      cv::rectangle(bin_img_yellow, cv::Rect(yellow_left, yellow_top, yellow_width, yellow_height), cv::Scalar(256/2), 2);
      // cv::imshow("bin_img_yellow", bin_img_yellow);
      if (isRedSignal)
      {
        if(num_labels_yellow > 1)
        {
          cv::rectangle(rgb, cv::Rect(left, top, width, height), cv::Scalar(0, 0, 255), 2); // 赤信号は赤い矩形
          cv::rectangle(top_region, cv::Rect(left, top, width, height), cv::Scalar(0, 0, 255), 2);
          red_light_flag = true;
        }
      }
      else
      {
        if(num_labels_yellow > 1)
        {
          cv::rectangle(rgb, cv::Rect(left, top, width, height), cv::Scalar(255, 0, 0), 2); // 青信号は青い矩形
          cv::rectangle(top_region, cv::Rect(left, top, width, height), cv::Scalar(255, 0, 0), 2);
          green_light_flag = true;
        }
      }
    }
  }
}

void drawOverlay(cv::Mat &image, bool red_light_flag, bool green_light_flag) 
{
  cv::Scalar color;
  if (red_light_flag) {
    color = cv::Scalar(0, 0, 255); // 赤色
    cv::rectangle(image, cv::Rect(10, 580, 60, 60), color, -1); // 赤信号の位置に赤い塗りつぶし矩形を描画
  } 
  if(green_light_flag) {
    color = cv::Scalar(255, 0, 0); // 青色
    cv::rectangle(image, cv::Rect(10, 650, 60, 60), color, -1); // 青信号の位置に青い塗りつぶし矩形を描画
  }
}

void addTextToImage(cv::Mat &image, const std::string &light_msg_state)
{
  int fontFace = cv::FONT_HERSHEY_SIMPLEX;
  double fontScale = 2;
  int thickness = 5;
  cv::Point textOrg(0, 0);
  if(light_msg_state=="RedLight")
  {
    cv::Point textOrg(80, 630); // 文字列を表示する位置
    cv::putText(image, light_msg_state, textOrg, fontFace, fontScale, cv::Scalar(0, 0, 255), thickness);
  } else if(light_msg_state=="GreenLight"){
    cv::Point textOrg(80, 700); // 文字列を表示する位置
    cv::putText(image, light_msg_state, textOrg, fontFace, fontScale, cv::Scalar(255, 0, 0), thickness);
  }
}

int main(int argc, char **argv)
{
  vector<fs::path> files_png;
  int file_cnt = 0;

  if(argc < 2){
    cout << "Usage is : " <<  argv[0] << "[image_directory_path]" << endl;
    cout << "困ったら/home/chiba/share/camera_lidar_data/img/ を入力" << endl;
    return -1;
  }
  /* ディレクトリからpngおよびpcdファイル名を取得 */
  /* files_pngおよびfiles_pcdにファイル名を格納 */
  getFiles(argv[1],".png",files_png);

  while (true)
  {
    cv::Mat camera_img = imread(files_png[file_cnt].string(), 1);

    if (camera_img.empty())
    {
      std::cerr << "Error: Could not read camera_img" << std::endl;
      break;
    }
    cout << "camera file : " << files_png[file_cnt].string() << endl;

    int height = camera_img.rows;
    int width = camera_img.cols;
    int top_region_height = height * IMAGE_ABOVE_RASIO;

    cv::Mat top_region = camera_img(cv::Rect(0, 0, width, top_region_height));

    cv::Mat hsv;
    cv::cvtColor(top_region, hsv, cv::COLOR_BGR2HSV);

    cv::Mat extract_red(top_region.size(), top_region.type(), cv::Scalar(0, 0, 0));
    cv::Mat extract_green(top_region.size(), top_region.type(), cv::Scalar(0, 0, 0));

    // カメラ画像から赤緑を抽出
    extractRedSignal(top_region, hsv, extract_red);
    extractGreenSignal(top_region, hsv, extract_green);

    // 二値化
    cv::Mat bin_img_red = cv::Mat::zeros(top_region.size(), CV_8UC1);
    cv::Mat bin_img_green = cv::Mat::zeros(top_region.size(), CV_8UC1);
    binalizeImage(extract_red, bin_img_red);
    binalizeImage(extract_green, bin_img_green);

    // メディアンフィルターにかける
    cv::Mat red_median(bin_img_red.size(), bin_img_red.type(), cv::Scalar(0));
    cv::Mat green_median(bin_img_green.size(), bin_img_green.type(), cv::Scalar(0));
    cv::medianBlur(bin_img_red, red_median, 5);
    cv::medianBlur(bin_img_green, green_median, 5);

    cv::imshow("green_median", green_median);

    // 膨張処理
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    Mat red_dilate = Mat::zeros(top_region.size(), CV_8UC1);
    Mat green_dilate = Mat::zeros(top_region.size(), CV_8UC1);
    cv::dilate(red_median, red_dilate, kernel);
    cv::dilate(green_median, green_dilate, kernel);

    // 候補領域のラベリング
    cv::Mat labeled_red, labeled_green;
    cv::Mat stats_red, stats_green, centroids_red, centroids_green;
    int num_labels_red, num_labels_green;

    num_labels_red = cv::connectedComponentsWithStats(red_dilate, labeled_red, stats_red, centroids_red);
    std::vector<int> red_width(num_labels_red), red_height(num_labels_red), red_left(num_labels_red), red_top(num_labels_red);
    for (int label = 1; label < num_labels_red; label++)
    {
      red_top[label] = stats_red.at<int>(label, cv::CC_STAT_TOP);
      red_left[label] = stats_red.at<int>(label, cv::CC_STAT_LEFT);
      red_width[label] = stats_red.at<int>(label, cv::CC_STAT_WIDTH);
      red_height[label] = stats_red.at<int>(label, cv::CC_STAT_HEIGHT);
      
      // ピンク色の矩形を描く
      // ピクセル数とアスペクト比を見る
      pixel_num = width * height;
      aspect_ratio = ((double)width)/((double)height);
      if(pixel_num<MIN_PIX_NUM || pixel_num>MAX_PIX_NUM)
      {
        continue;
      }
      if(aspect_ratio<MIN_ASPECT_RATIO || aspect_ratio>MAX_ASPECT_RATIO)
      {
        continue;
      }
      cv::rectangle(top_region, cv::Rect(red_left[label], red_top[label], red_width[label], red_height[label]), cv::Scalar(255,0,255), 2);
    }

    num_labels_green = cv::connectedComponentsWithStats(green_dilate, labeled_green, stats_green, centroids_green);
    std::vector<int> green_width(num_labels_green), green_height(num_labels_green), green_left(num_labels_green), green_top(num_labels_green);
    for (int label = 1; label < num_labels_green; label++)
    {
      green_top[label] = stats_green.at<int>(label, cv::CC_STAT_TOP);
      green_left[label] = stats_green.at<int>(label, cv::CC_STAT_LEFT);
      green_width[label] = stats_green.at<int>(label, cv::CC_STAT_WIDTH);
      green_height[label] = stats_green.at<int>(label, cv::CC_STAT_HEIGHT);

      // 水色の矩形を描く
      // ピクセル数、アスペクト比を見る
      pixel_num = width * height;
      aspect_ratio = ((double)width) / ((double)height);
      if(pixel_num < MIN_PIX_NUM || pixel_num > MAX_PIX_NUM)
      {
        continue;
      }
      if(aspect_ratio < MIN_ASPECT_RATIO || aspect_ratio > MAX_ASPECT_RATIO)
      {
        continue;
      }
      cv::rectangle(camera_img, cv::Rect(green_left[label], green_top[label], green_width[label], green_height[label]), cv::Scalar(255,255,0), 2);
      cv::rectangle(top_region, cv::Rect(green_left[label], green_top[label], green_width[label], green_height[label]), cv::Scalar(255,255,0), 2);
    }

    extractYellowInBlob(camera_img, red_dilate, num_labels_red, red_width, red_height, red_left, red_top, true, top_region);
    extractYellowInBlob(camera_img, green_dilate, num_labels_green, green_width, green_height, green_left, green_top, false,top_region);

    // 赤、青信号が連続で検出されるほどcountが加算されていく
    // red_light_flag, greem_light_flagはextractYellowBlob関数から出力されている
    if(red_light_flag) red_cnt++;
    else red_cnt = 0;

    if(green_light_flag) green_cnt++;
    else green_cnt = 0;

    if(red_cnt>RED_IMAGE_THRESH)
    {
      drawOverlay(camera_img, red_light_flag, green_light_flag);
      light_msg_state = "RedLight";
      addTextToImage(camera_img, light_msg_state);
      red_cnt=0;
    }
    if(green_cnt>GREEN_IMAGE_THRESH)
    {
      drawOverlay(camera_img, red_light_flag, green_light_flag);
      light_msg_state = "GreenLight";
      addTextToImage(camera_img, light_msg_state);
      green_cnt = 0;
    }

    red_light_flag = false;
    green_light_flag = false;

    cv::imshow("Result", camera_img);

    int key = cv::waitKey(0);
    if(key == ' ') break;
    else if(key == 'a') file_cnt--;
    else if(key == 'A') file_cnt -= 10;
    else if(key == 'D') file_cnt += 10;
    else if(key == 'd') file_cnt++;
    if(file_cnt < 0) file_cnt = 0;
  }
    return 0;
}