using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

using UnityEngine;

using static UnityEditor.Experimental.GraphView.GraphView;

public class MLPParameters
{
    List<float[,]> coeficients;
    List<float[]> intercepts;
    int layers;

    public MLPParameters(int numLayers)
    {
        layers = numLayers;

        coeficients = new List<float[,]>();
        intercepts = new List<float[]>();
        for (int i = 0; i < numLayers - 1; i++)
        {
            coeficients.Add(null);
        }
        for (int i = 0; i < numLayers - 1; i++)
        {
            intercepts.Add(null);
        }
    }

    public void CreateCoeficient(int i, int rows, int cols)
    {
        coeficients[i] = new float[rows, cols];
    }

    public void SetCoeficiente(int i, int row, int col, float v)
    {
        coeficients[i][row, col] = v;
    }

	public List<float[,]> GetCoeficientList() {
		return coeficients;
	}

	public void CreateIntercept(int i, int row)
    {
        intercepts[i] = new float[row];
    }

    public void SetIntercept(int i, int row, float v)
    {
        intercepts[i][row] = v;
    }

	public float[] GetIntercept(int i) {
		return intercepts[i];
	}

	public int GetLayers() {
		return layers;
	}
}

public class MLPModel
{
    MLPParameters mlpParameters;
    public MLPModel(MLPParameters p)
    {
        mlpParameters = p;
    }

    /// <summary>
    /// Parameters required for model input. By default it will be perception, kart position and time, 
    /// but depending on the data cleaning and data acquisition modificiations made by each one, the input will need more parameters.
    /// </summary>
    /// <param name="p">The Agent perception</param>
    /// <returns>The action label</returns>
    public float[] FeedForward(Perception p, Transform transform)
    {
        Parameters parameters = Record.ReadParameters(8, Time.timeSinceLevelLoad, p, transform);
        float[] input=parameters.ConvertToFloatArrat();

        //TODO: implement feedworward.
        //the size of the output layer depends on what actions you have performed in the game.
        //By default it is 7 (number of possible actions) but some actions may not have been performed and therefore the model has assumed that they do not exist.
        List<float[]> activations = new List<float[]>();

        const int numRays = 5;
        activations.Add(new float[numRays]);
        for (int i = 0; i < numRays; i++) {
			if (input[i] == -1)
				activations[0][i] = 1;
			else
				activations[0][i] = input[i] / p.distance[i];
		}
            
		activations[0] = AddOnesColumn(activations[0]);

		List<float[,]> thetaList = mlpParameters.GetCoeficientList();

        for (int k = 0; k < thetaList.Count; k++) {
            float[] z = new float[activations[k].Length - 1];

            for (int i = 0; i < z.Length; i++) {
                z[i] = 0;
                for (int j = 0; j < z.Length; j++) {
                    z[i] += activations[k][i] * thetaList[k][j, i];
                }
            }

			if (k < thetaList.Count - 1) {
                activations[k] = AddOnesColumn(activations[k]);

				activations.Add(new float[thetaList[k + 1].GetLength(1)]);

				for (int i = 0; i < thetaList[k + 1].GetLength(1); i++) {
					activations[k + 1][i] = Sigmoid(z[i]);
				}
			}
		}

		return activations[activations.Count - 1];
	}

	private float[] AddOnesColumn(float[] vector) {
		int length = vector.Length;
		float[] result = new float[length + 1];
		result[0] = 1.0f; // Bias term

		for (int i = 1; i <= length; i++) {
			result[i] = vector[i - 1];
		}

		return result;
	}

	float Sigmoid(float z) {
		return 1 / (1 + Mathf.Exp(-z));
	}

	/// <summary>
	/// Implements the conversion of the output value to the action label. 
	/// Depending on what actions you have chosen or saved in the dataset, and in what order, the way it is converted will be one or the other.
	/// </summary>
	/// <param name="index"></param>
	/// <returns></returns>
	public Labels ConvertIndexToLabel(int index)
    {
		switch (index) {
			case 0:
				return Labels.NONE;
			case 1:
				return Labels.ACCELERATE;
			case 2:
				return Labels.LEFT_ACCELERATE;
			case 3:
				return Labels.RIGHT_ACCELERATE;
			default:
				return Labels.NONE;
		}
	}

    public Labels Predict(float[] output)
    {
        int index = GetIndexMaxValue(output, out float max);
        Debug.Log(index + " " + max);
        Labels label = ConvertIndexToLabel(index);
        return label;
    }

    public int GetIndexMaxValue(float[] output, out float max)
    {
        max = output[0];
        int index = 0;
        for(int i = 1; i < output.Length; i++)
        {
            if(output[i] > max)
            {
                max = output[i];
                index = i;
            }
        }
        return index;
    }
}

public class MLAgent : MonoBehaviour
{
    public enum ModelType { MLP=0 }
    public TextAsset text;
    public ModelType model;
    public bool agentEnable;

    private MLPParameters mlpParameters;
    private MLPModel mlpModel;
    private Perception perception;

    // Start is called before the first frame update
    void Start()
    {
        if (agentEnable)
        {
            string file = text.text;
            if (model == ModelType.MLP)
            {
                mlpParameters = LoadParameters(file);
                mlpModel = new MLPModel(mlpParameters);
            }
            Debug.Log("Parameters loaded " + mlpParameters);
            perception = GetComponent<Perception>();
        }
    }



    public KartGame.KartSystems.InputData AgentInput()
    {
        Labels label = Labels.NONE;
        switch (model)
        {
            case ModelType.MLP:
                float[] outputs = this.mlpModel.FeedForward(perception,this.transform);
                label = this.mlpModel.Predict(outputs);

                Debug.Log(label);
                break;
        }
        KartGame.KartSystems.InputData input = Record.ConvertLabelToInput(label);
        return input;
    }

    public static string TrimpBrackers(string val)
    {
        val = val.Trim();
        val = val.Substring(1);
        val = val.Substring(0, val.Length - 1);
        return val;
    }

    public static int[] SplitWithColumInt(string val)
    {
        val = val.Trim();
        string[] values =val.Split(",");
        int[] result = new int[values.Length];
        for(int i = 0; i < values.Length; i++)
        {
            values[i] = values[i].Trim();
            if (values[i].StartsWith("'"))
                values[i] = values[i].Substring(1);
            if (values[i].EndsWith("'"))
                values[i] = values[i].Substring(0, values[i].Length-1);
            result[i] = int.Parse(values[i]);
        }
        return result;
    }

    public static float[] SplitWithColumFloat(string val)
    {
        val = val.Trim();
        string[] values = val.Split(",");
        float[] result = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = values[i].Replace('.', ',');
            result[i] = float.Parse(values[i]);
        }
        return result;
    }

    public static MLPParameters LoadParameters(string file)
    {
        string[] lines = file.Split("\n");
        int num_layers = 0;
        MLPParameters mlpParameters = null;
        int currentParameter = -1;
        int[] currentDimension = null;
		bool coefficient = false, intercept = false;
		for (int i = 0; i < lines.Length; i++)
        {
			

			string line = lines[i];
            line = line.Trim();
            if(line != "")
            {
                string[] nameValue = line.Split(":");
                string name = nameValue[0];
                string val = nameValue[1];
                if (name == "num_layers")
                {
                    num_layers = int.Parse(val);
                    mlpParameters = new MLPParameters(num_layers);
                }
                else
                {
                    if (num_layers <= 0)
                        Debug.LogError("Format error: First line must be num_layers");
                    else
                    {
                        if (name == "parameter")
                            currentParameter = int.Parse(val);
                        else if (name == "dims")
                        {
                            val = TrimpBrackers(val);
                            currentDimension = SplitWithColumInt(val);
                        }
                        else if (name == "name")
                        {
                            if (val.StartsWith("coefficient"))
                            {
                                coefficient = true;
                                intercept = false;
                                int index = currentParameter / 2;
                                mlpParameters.CreateCoeficient(currentParameter, currentDimension[0], currentDimension[1]);
                            }
                            else if (val.StartsWith("intercepts")) {
                                intercept = true;
								coefficient = false;
								mlpParameters.CreateIntercept(currentParameter, currentDimension[1]);
                            }
                            else {
								coefficient = false;
								intercept = false;
							}
                        }
                        else if (name == "values")
                        {
                            val = TrimpBrackers(val);
                            float[] parameters = SplitWithColumFloat(val);

                            for (int index = 0; index < parameters.Length; index++)
                            {
                                if (coefficient)
                                {
                                    int row = index / currentDimension[1];
                                    int col = index % currentDimension[1];
                                    mlpParameters.SetCoeficiente(currentParameter, row, col, parameters[index]);
                                }
                                else if (intercept)
                                {
                                    mlpParameters.SetIntercept(currentParameter, index, parameters[index]);
								}
                            }
                        }
                    }
                }
            }
        }
        return mlpParameters;
    }
}
