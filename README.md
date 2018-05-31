# Optical Calibration Tool for Projector-Camera System

## Abstract
This program is for calibration optical response function between projector and camera.
The program generates projection patterns and calculates parameters a, b, gamma, and k.
You can modify your projection pattern's intensities by

> x' = ((kx)^(1/gamma) - b) / a,

where x is your original projection pattern's intensity and x' is modified projection pattern's intensity.
By using the modified intensities, the projector-camera system optical response function becomes linear.

## License

The license of these programs is MIT License.
Please use follows as your references.

> Naoya Chiba and Koichi Hashimoto. "Development of Calibration Tool for Projector-Camera System Considering Saturation And Gamma Correction." The Robotics and Mechatronics Conference 2018 (ROBOMECH 2018), 2A1-M13, Kitakyushu, June, 5th, 2018.

> 千葉 直也，橋本 浩一．サチュレーションとガンマ補正を考慮したプロジェクタ・カメラシステムの光学キャリブレーションツールの開発．ロボティクス・メカトロニクス講演会2018 (ROBOMECH 2018)，2A1-M13，北九州，6月5日，2018．
