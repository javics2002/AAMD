using System.Collections;
using System.Collections.Generic;
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

	public float[,] GetCoeficient(int index) {
		return coeficients[index];
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
		List<float> activation = new List<float>();
		activation.AddRange(input);

        //Limpiar el input de valores no usados por el perceptron
        activation.RemoveRange(5, 4);

		for (int c = 0; c < mlpParameters.GetLayers() - 1; c++) {
			float[,] coefficient = mlpParameters.GetCoeficient(c);
			float[] intercepts = mlpParameters.GetIntercept(c);
			float[] z = new float[coefficient.GetLength(1)];

			for (int i = 0; i < z.Length; i++) {
				z[i] = intercepts[i];
				for (int j = 0; j < activation.Count; j++) {
					z[i] += activation[j] * coefficient[j, i];
				}
			}

			activation.Clear();
			for (int i = 0; i < z.Length; i++) {
				activation.Add(Sigmoid(z[i]));
			}
		}

		return activation.ToArray();
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
        
        for (int i = 0; i < lines.Length; i++)
        {
			bool coefficient = false, intercept = false;

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
                                int index = currentParameter / 2;
                                mlpParameters.CreateCoeficient(currentParameter, currentDimension[0], currentDimension[1]);
                            }
                            else if (val.StartsWith("intercepts")) {
                                intercept = true;
                                mlpParameters.CreateIntercept(currentParameter, currentDimension[1]);
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
