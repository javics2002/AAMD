from skl2onnx import to_onnx
from onnx2json import convert
import pickle
import json

def IsParameterValid(parameter, parameter_name):
    if(parameter == 0):
        return parameter_name == "coefficient" or parameter_name == "intercepts"
    else:
        return parameter_name == f"coefficient{parameter}" or parameter_name == f"intercepts{parameter}"

def ExportONNX_JSON_TO_Custom(onnx_json,mlp):
    graphDic = onnx_json["graph"]
    initializer = graphDic["initializer"]
    s = "num_layers:"+str(mlp.n_layers_)+"\n"
    index = 0
    parameterIndex = 0
    for parameter in initializer:
        if not IsParameterValid(parameterIndex, parameter["name"]):
           continue

        s += "parameter:"+str(parameterIndex)+"\n"
        s += "dims:"+str(parameter["dims"])+"\n"
        s += "name:"+str(parameter["name"])+"\n"
        if "doubleData" in parameter:
            s += "values:"+str(parameter["doubleData"])+"\n"
        if "int32Data" in parameter:
            s += "values:"+str(parameter["int32Data"])+"\n"
        if "int64Data" in parameter:
            s += "values:"+str(parameter["int64Data"])+"\n"
        index = index + 1
        parameterIndex = index // 2
    return s

def ExportAllformatsMLPSKlearn(mlp,X,picklefileName,onixFileName,jsonFileName,customFileName):
    with open(picklefileName,'wb') as f:
        pickle.dump(mlp,f)
    
    onx = to_onnx(mlp, X[:1])
    with open(onixFileName, "wb") as f:
        f.write(onx.SerializeToString())
    
    onnx_json = convert(input_onnx_file_path=onixFileName,output_json_path=jsonFileName,json_indent=2)
    
    customFormat = ExportONNX_JSON_TO_Custom(onnx_json,mlp)
    with open(customFileName, 'w') as f:
        f.write(customFormat)
