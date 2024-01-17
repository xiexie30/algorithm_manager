#!/bin/bash

# trtexec --onnx=workspace/yolov5s.onnx \
#     --minShapes=images:1x3x640x640 \
#     --maxShapes=images:16x3x640x640 \
#     --optShapes=images:1x3x640x640 \
#     --saveEngine=workspace/yolov5s.engine


# trtexec --onnx=workspace/yolov7-s2-fp16.onnx \
#     --inputIOFormats=fp16:chw \
#     --outputIOFormats=fp16:chw \
#     --workspace=1024 \
#     --fp16 \
#     --saveEngine=workspace/yolov7-s2-fp16.engine

# trtexec --onnx=workspace/yolov8m2.transd.onnx \
#    --fp16 \
#    --saveEngine=workspace/yolov8m2-fp16.engine
   # --verbose

trtexec --onnx=workspace/yolov8m-640-fp16.transd.onnx \
   --fp16 \
   --saveEngine=workspace/yolov8m-640-fp16.engine
   # --verbose 

# trtexec --onnx=workspace/prune-yolov8m-p234-SEAM-dyhead-simAM-fasterSimAM.onnx \
#    --inputIOFormats=fp16:chw \
#    --outputIOFormats=fp16:chw \
#    --workspace=1024 \
#    --fp16 \
#    --plugins=build/ScatterND.so \
#    --plugins=build/MMCVModulatedDeformConv2d.so \
#    --saveEngine=workspace/prune-yolov8m-p234-SEAM-dyhead-simAM-fasterSimAM-fp16.engine \
#    --verbose 
    
# trtexec --onnx=workspace/prune-db-yolov8m-640.transd.onnx \
#     --minShapes=images:1x3x640x640 \
#     --maxShapes=images:16x3x640x640 \
#     --optShapes=images:1x3x640x640 \
#     --saveEngine=workspace/prune-db-yolov8m-640-fp16.engine \
#     --workspace=4096 \
#     --fp16
#    --inputIOFormats=fp16:chw \
#    --outputIOFormats=fp16:chw

# trtexec --onnx=workspace/yolov8n.transd.onnx \
#     --saveEngine=workspace/yolov8n-2.engine

# trtexec --onnx=workspace/yolov8n.transd.onnx \
#     --minShapes=images:1x3x640x640 \
#     --maxShapes=images:16x3x640x640 \
#     --optShapes=images:1x3x640x640 \
#     --saveEngine=workspace/yolov8n.engine

# trtexec --onnx=workspace/yolov8n-seg.b1.transd.onnx \
#     --saveEngine=workspace/yolov8n-seg.b1.transd.engine