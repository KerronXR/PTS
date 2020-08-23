package com.example.potholetrackingsystem;

import org.opencv.core.*;

import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.Core.bitwise_and;
import static org.opencv.imgproc.Imgproc.*;

public class OpenCVUtils {

    public Mat region_of_interest(Mat img, List<MatOfPoint> vertices) {
        Mat mask = new Mat(img.size(0), img.size(1), CvType.CV_8U, Scalar.all(0));
        Scalar match_mask_color = new Scalar(255);
        Mat masked_image = new Mat();
        fillPoly(mask, vertices, match_mask_color);
        bitwise_and(img, mask, masked_image);
        return masked_image;
    }

    public List<MatOfPoint> elementToPoints(double[][] points_to_add) {
        List<Point> points = new ArrayList<>();
        for (int i = 0; i < points_to_add.length; i++) {
            points.add(new Point(points_to_add[i]));
        }
        MatOfPoint mPoints = new MatOfPoint();
        mPoints.fromList(points);
        List<MatOfPoint> finalPoints = new ArrayList<MatOfPoint>();
        finalPoints.add(mPoints);
        return finalPoints;
    }

    public Mat[] find_road_area(Mat cropped_frame, Mat original_frame, Mat grey_copy) {
        Mat[] toReturn = new Mat[2];
        Mat lines = new Mat();
        Mat roi_frame = new Mat(original_frame.size(0), original_frame.size(1), CvType.CV_8U, Scalar.all(0));
        double x1, x2, y1, y2, m, b, mx, cross_x, cross_y;
        HoughLinesP(cropped_frame, lines, 1, 3.14159265359 / 180, 35, 80, 200);
        double[] line_left = {0, 0, 0, 0, 0, 0};
        double[] line_right = {0, 0, 0, 0, 0, 0};
        double[][] roi_vertices = {{0, 0}, {0, 0}, {0, 0}};
        boolean found_left = false;
        boolean found_right = false;
        if (!lines.empty()) {
            for (int i = 0; i < lines.rows(); i++) {
                x1 = (lines.get(i, 0)[0]);
                y1 = (lines.get(i, 0)[1]);
                x2 = (lines.get(i, 0)[2]);
                y2 = (lines.get(i, 0)[3]);
                m = ((y2 - y1) / (x2 - x1));
                if (0.7 > m && m > 0.3 && !found_right) {
                    found_right = true;
                    b = y1 - (m * x1);
                    line_right[0] = x1;
                    line_right[1] = y1;
                    line_right[2] = x2 + 10;
                    line_right[3] = y2;
                    line_right[4] = m;
                    line_right[5] = b;
                    line(original_frame, new Point(x1, y1), new Point(x2, y2), new Scalar(0, 255, 0), 5);
                    if (found_left) {
                        break;
                    }
                }
                if (-0.7 < m && m < -0.3 && !found_left) {
                    found_left = true;
                    b = y1 - (m * x1);
                    line_left[0] = x1 + 20;
                    line_left[1] = y1;
                    line_left[2] = x2;
                    line_left[3] = y2;
                    line_left[4] = m;
                    line_left[5] = b;
                    line(original_frame, new Point(x1, y1), new Point(x2, y2), new Scalar(0, 255, 0), 5);
                    if (found_right) {
                        break;
                    }
                }
            }
        }
        if (found_right && found_left) {
            if (line_left != line_right) {
                mx = line_right[4] - line_left[4];
                if (mx != 0) {
                    b = line_left[5] - line_right[5];
                    if (b == 0) {
                        cross_x = 0;

                    } else {
                        cross_x = b / mx;
                    }
                    cross_y = line_right[4] * cross_x + line_right[5];
                    roi_vertices[0][0] = (int) line_left[0];
                    roi_vertices[0][1] = (int) line_left[1];
                    roi_vertices[1][0] = (int) cross_x;
                    roi_vertices[1][1] = (int) cross_y;
                    roi_vertices[2][0] = (int) line_right[2];
                    roi_vertices[2][1] = (int) line_right[3];
                    roi_frame = region_of_interest(grey_copy, elementToPoints(roi_vertices));
                }
            }
        } else {
            putText(original_frame, "Lane Not Found",
                    new Point(original_frame.size(0) / 3.0, original_frame.size(1) / 2.0), FONT_HERSHEY_SIMPLEX,
                    1.5, new Scalar(255, 0, 0), 4, LINE_AA);
        }
        toReturn[0] = roi_frame;
        toReturn[1] = original_frame;
        return toReturn;
    }

}
