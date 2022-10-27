//#define TestingPointInTriangle // <- draws random triangles, and checks every point to see if it is in the triangle (think of inefficient triangle fill, but good to test method)
using Microsoft.VisualBasic.Devices;
using SheepHerderMutation;
using SheepHerderMutation.AI;
using SheepHerderMutation.Configuration;
using SheepHerderMutation.Utilities;
using System;
using System.Diagnostics;
using System.Drawing.Drawing2D;
using System.Security.Cryptography;

namespace SheepHerderMutation;

/// <summary>
/// Represents the simple form of "Images" (containing flock+dog) or graphs.
/// </summary>
public partial class Form1 : Form
{
    /// <summary>
    /// Used to control the timer.
    /// </summary>
    internal static System.Windows.Forms.Timer s_timer = new();

    /// <summary>
    /// Size of each play area (small enough to show lots on a 17" monitor whilst still working effectively.
    /// </summary>
    const int c_width = 300;

    /// <summary>
    /// Size of each play area (small enough to show lots on a 17" monitor whilst still working effectively.
    /// </summary>
    const int c_height = 300;
   
    /// <summary>
    /// This stores the images (one per neural network id / flock).
    /// </summary>
    readonly PictureBox[] pictureBoxArray;

    #region EVENT HANDLERS
    /// <summary>
    /// Constructor.
    /// </summary>
    public Form1()
    {
        InitializeComponent();
        s_timer.Interval = 5;

        List<PictureBox> pictureBoxes = new();
        
        for (int i = 0; i < Config.NumberOfAIdogs; i++)
        {
            PictureBox pb = new()
            {
                Size = new Size(c_width, c_height),
                SizeMode = PictureBoxSizeMode.StretchImage,
                Margin = new(1),
                Tag = i
            };
        
            flowLayoutPanel1.Controls.Add(pb);
            
            //pb.MouseMove += new System.Windows.Forms.MouseEventHandler(this.PictureBox1_MouseMove);
            pb.MouseDown += new System.Windows.Forms.MouseEventHandler(this.PictureBox1_MouseDown);

            pictureBoxes.Add(pb);
        }

        pictureBoxArray = pictureBoxes.ToArray();
        LearnToHerd.WhenMutating += LearnToHerd_WhenMutating;
    }

    /// <summary>
    /// After mutation, in silent mode we draw graphs of performance per neural network.
    /// </summary>
    private void LearnToHerd_WhenMutating()
    {
        Text = $"SheepHerderAI | Generation: {LearnToHerd.s_generation}";
        GraphPerformance();
    }

    /// <summary>
    /// Animation is frame by frame, initiated each tick.
    /// </summary>
    /// <param name="sender"></param>
    /// <param name="e"></param>
    private void Timer1_Tick(object? sender, EventArgs e)
    {
        //if (flock == null) throw new Exception("please initialise the flock in Onload()");
        
        LearnToHerd.Learn();
        List<Bitmap> images = LearnToHerd.DrawAll(out bool mutateNow);

        if (mutateNow)
        {
            s_timer.Stop();
            LearnToHerd.NextGeneration();
            LearnToHerd_WhenMutating();
            s_timer.Start();
            return;
        }
        
        if (LearnToHerd.s_silentMode) return; // no image to draw

        int i = 0;

        foreach (Bitmap image in images)
        {
            pictureBoxArray[i].Image?.Dispose();
            pictureBoxArray[i].Image = image;
            ++i;
        }
    }

    /// <summary>
    /// Draws a "performance" graph for each neural network.
    /// </summary>
    private void GraphPerformance()
    {
        if (!LearnToHerd.s_silentMode) return; // graphs are drawn in silent mode

        const bool c_drawUsingParallelThreads = true;

        // id, score
        Dictionary<int,int> scores = new();
        
        foreach (NeuralNetwork n in NeuralNetwork.s_networks.Values) scores.Add(n.Id,(int) Math.Round(n.Score));

        //https://stackoverflow.com/questions/66349874/how-to-rank-elements-in-c-sharp-especially-when-it-has-duplicates
        // var ranks = input.OrderByDescending(x => x).ToArray();
        // var ranked = input.Select(x => Array.IndexOf(ranks, x) + 1);

        var rankLookup = scores.Values.ToArray()
                            .OrderByDescending(i => i)
                            .Select((num, index) => (num, index))
                            .ToLookup(x => x.num, x => x.index + 1);

        int[] result = scores.Values.Select(i => rankLookup[i].First()).ToArray();

        int x = 0;
        
        foreach(int id in scores.Keys)
        {
            NeuralNetwork.s_networks[id].Rank = result[x++];
        }

#pragma warning disable CS0162 // Unreachable code detected - config drives whether parallel or not parallel, so both need to exist
        if (c_drawUsingParallelThreads)
        {
            // all sheep are independent, each one has a neural network and sensors attached, this therefore
            // is a candidate for parallelism.
            Parallel.ForEach(LearnToHerd.s_flock.Values, flock =>
            {
                DrawGraphForNetwork(flock.Id);
            });
        }
        else
        {
            for (int neuralNetworkId = 0; neuralNetworkId < LearnToHerd.s_flock.Count; neuralNetworkId++)
            {
                DrawGraphForNetwork(neuralNetworkId);
            }
#pragma warning restore CS0162 // Unreachable code detected
        }
    }

    /// <summary>
    /// Draw the graph and assigns to the picture box.
    /// </summary>
    /// <param name="neuralNetworkId"></param>
    void DrawGraphForNetwork(int neuralNetworkId)
    {
        Bitmap image = Graph.CreateSimpleImageOffPerformanceValues(NeuralNetwork.s_networks[neuralNetworkId].Rank, NeuralNetwork.s_networks[neuralNetworkId].Performance, neuralNetworkId, c_width, c_height);
        pictureBoxArray[neuralNetworkId].Image?.Dispose();
        pictureBoxArray[neuralNetworkId].Image = image;
    }

    /// <summary>
    /// On form load, we create the flock, and position fences and score zone.
    /// </summary>
    /// <param name="sender"></param>
    /// <param name="e"></param>
    private void Form1_Load(object sender, EventArgs e)
    {
#if TestingPointInTriangle
        s_timer.Interval = 1000;
        s_timer.Tick += Timer1_Triangles;
        s_timer.Start();
        return;
#endif

        LearnToHerd.s_sheepPenScoringZone = new Rectangle(c_width - (c_width / 7), 0, c_width / 7, c_height / 7);

        DefineSheepPenAndFences();

        LearnToHerd.StartLearning(c_width, c_height);

        s_timer.Tick += Timer1_Tick;
        s_timer.Start();
    }

    /// <summary>
    /// A very basic set of lines (baffles) to make the AI work herding.
    /// </summary>
    private static void DefineSheepPenAndFences()
    {
        List<PointF[]> linesToDraw = new();

        // the pen
        List<PointF> lines = new()
        {
            new PointF(LearnToHerd.s_sheepPenScoringZone.Right-1, LearnToHerd.s_sheepPenScoringZone.Height),
            new PointF(LearnToHerd.s_sheepPenScoringZone.Right-1, 0),
            new PointF(LearnToHerd.s_sheepPenScoringZone.Left, 0),
            new PointF(LearnToHerd.s_sheepPenScoringZone.Left, LearnToHerd.s_sheepPenScoringZone.Height),
            new PointF(LearnToHerd.s_sheepPenScoringZone.Left-LearnToHerd.s_sheepPenScoringZone.Width/2, LearnToHerd.s_sheepPenScoringZone.Height+LearnToHerd.s_sheepPenScoringZone.Height/2)
        };

        linesToDraw.Add(lines.ToArray());

        // all the way around the screen
        lines = new()
        {
            new PointF(2, 2),
            new PointF(c_width-3, 2),
            new PointF(c_width-3, c_height-3),
            new PointF(2, c_height-3),
            new PointF(2, 2)
        };

        linesToDraw.Add(lines.ToArray());

        // the start
        lines = new()
        {
            new PointF(c_width / 4, 0),
            new PointF(c_width/ 4, c_height/ 4 * 3)
        };

        linesToDraw.Add(lines.ToArray());

        // a restricted point
        lines = new()
        {
            new PointF(c_width / 2, 0),
            new PointF(c_width / 2, c_height/ 4 *1.8f),
            new PointF(c_width/2 + c_width / 4, c_height / 4 * 1.8f)
        };
        linesToDraw.Add(lines.ToArray());

        lines = new()
        {
            new PointF(c_width / 2, c_height),
            new PointF(c_width / 2, c_height - c_height/ 4 * 1.8f)
        };
        linesToDraw.Add(lines.ToArray());

        LearnToHerd.s_lines = linesToDraw;


        // these are "way points" that the sheep must go thru, that the AI must try to make happen
        // Points were based on 674 x 500, so we need to scale them here.
        /*
        LearnToHerd.s_wayPointsSheepNeedsToGoThru = new Point[] {
                                    // point 0 must be lower than all first sheep
                                    ScalePoint(83, 113), //0
                                    ScalePoint(83, 143), //1
                                    ScalePoint(81, 192), //2
                                    ScalePoint(79, 241), //3
                                    ScalePoint(78, 281), //4
                                    ScalePoint(78, 321), //5
                                    ScalePoint(87, 357), //6                                    
                                    ScalePoint(96, 393), //7
                                    ScalePoint(121, 414),//8
                                    ScalePoint(165, 434),//9
                                    ScalePoint(200, 415),//10
                                    ScalePoint(230, 391),//11
                                    ScalePoint(251, 368),//12
                                    ScalePoint(271, 322),//13
                                    ScalePoint(287, 291),//14
                                    ScalePoint(316, 260),//15
                                    ScalePoint(364, 255),//16
                                    ScalePoint(410, 249),//17
                                    ScalePoint(463, 249),//18
                                    ScalePoint(517, 250),//19
                                    ScalePoint(543, 228), //20
                                    ScalePoint(570, 207), //21
                                    ScalePoint(586, 178), //22
                                    ScalePoint(602, 146), //23
                                    ScalePoint(612, 93), //24
                                    ScalePoint(619, 40) //25
        };
        */
        LearnToHerd.s_wayPointsSheepNeedsToGoThru = new Point[] {
                                // point 0 must be lower than all first sheep
                               // ScalePoint(83, 113),  //0
                               // ScalePoint(83, 143),  //1
                                ScalePoint(81, 192),  //2
                              //  ScalePoint(79, 241),  //3
                              //  ScalePoint(78, 281),  //4
                                ScalePoint(78, 321),  //5
                              //  ScalePoint(87, 357),  //6                                    
                              //  ScalePoint(96, 393),  //7
                                ScalePoint(155, 427), //8
                              //  ScalePoint(165, 434), //9
                              //  ScalePoint(200, 415), //10
                              //  ScalePoint(230, 391), //11
                              //  ScalePoint(251, 368), //12
                              //  ScalePoint(271, 322), //13
                                ScalePoint(310, 260), //14
                                ScalePoint(336, 260), //15
                              //  ScalePoint(364, 255), //16
                              //  ScalePoint(410, 249), //17
                              //  ScalePoint(463, 249), //18
                              //  ScalePoint(517, 250), //19
                                ScalePoint(555, 260), //20
                              //  ScalePoint(570, 207), //21
                              //  ScalePoint(586, 178), //22
                                ScalePoint(602, 146), //23
                              //  ScalePoint(612, 93),  //24
                                ScalePoint(619, 40)   //25
        };
  
        /*
        // sanity check, to ensure the gap between points is not too large based on config max deviation
        for (int i = 0; i < LearnToHerd.s_wayPointsSheepNeedsToGoThru.Length - 1; i++)
        {
            float dist = MathUtils.DistanceBetweenTwoPoints(LearnToHerd.s_wayPointsSheepNeedsToGoThru[i], LearnToHerd.s_wayPointsSheepNeedsToGoThru[i + 1]);
            
            if (dist > Config.SheepMaxDeviationFromWayPoint)
            {
                Debug.WriteLine($"{dist}   {LearnToHerd.s_wayPointsSheepNeedsToGoThru[i]} to {LearnToHerd.s_wayPointsSheepNeedsToGoThru[i + 1]}");
                throw new Exception("way points are spaced too far apart");
            }
        }
        */
    }

    /// <summary>
    /// The points were based on a 674x500 grid. This makes them proportionate to the size of the picturebox.
    /// </summary>
    /// <param name="x"></param>
    /// <param name="y"></param>
    /// <returns></returns>
    private static Point ScalePoint(int x, int y)
    {
        return new Point((int)((float)x / 674f * c_width), (int)((float)y / 500f * c_height));
    }

#if TestingPointInTriangle
 
    /// <summary>
    /// Proof the .PtInTriangle() works, and correctly detects pixels that are within a triangle.
    /// </summary>   
    private void Timer1_Triangles(object? sender, EventArgs e)
    {
        // pick 3 random points, that make up our "triangle"
        PointF p1 = new(RandomNumberGenerator.GetInt32(0, pictureBox1.Width), RandomNumberGenerator.GetInt32(0, pictureBox1.Height));

        PointF p2 = new(RndomNumberGenerator.GetInt32(0, pictureBox1.Width), RandomNumberGenerator.GetInt32(0, pictureBox1.Height));

        PointF p3 = new(RandomNumberGenerator.GetInt32(0, pictureBox1.Width), RandomNumberGenerator.GetInt32(0, pictureBox1.Height));

        Bitmap b = new Bitmap(pictureBox1.Width, pictureBox1.Height);
        using Graphics graphics = Graphics.FromImage(b);

        // draw the triangle.
        graphics.DrawLines(Pens.Red, new PointF[] { p1, p2, p3 });

        graphics.Flush();

        // for every pixel (width*height), plot "black" if inside the triangle
        for (int x=0;x<pictureBox1.Width;x++)
        {
            for(int y=0;y<pictureBox1.Height;y++)
            {
                if(Utils.PtInTriangle(new PointF(x,y),p1,p2,p3))
                {
                    b.SetPixel(x, y, Color.Black);
                }
            }
        }

        graphics.Flush();

        pictureBox1.Image?.Dispose();
        pictureBox1.Image = b;
    }
#endif

    #endregion

    /// <summary>
    /// User can press keys to save/load model, pause/slow, mutate etc.
    /// </summary>
    /// <param name="sender"></param>
    /// <param name="e"></param>
    private void Form1_KeyDown(object sender, KeyEventArgs e)
    {
        switch (e.KeyCode)
        {
            case Keys.P:
                s_timer.Enabled = !s_timer.Enabled;
                break;

            case Keys.F:
                // "F" slow mode
                StepThroughSpeeds();
                break;

            case Keys.S:
                // "S" saves the neural network
                NeuralNetwork.Save();
                break;

            case Keys.Q:
                // "Q" quiet mode
                LearnToHerd.s_silentMode = !LearnToHerd.s_silentMode;
                GraphPerformance(); // force paint of graph
                break;

        }
    }

    /// <summary>
    /// Pressing "S" slows things down, 2x slower, 5x slower, 10x slower, then back to normal speed.
    /// </summary>
    internal static void StepThroughSpeeds()
    {
        var newInterval = s_timer.Interval switch
        {
            10 => 20,
            20 => 100,
            100 => 1000,
            _ => 10,
        };

        s_timer.Interval = newInterval;
    }

    /// <summary>
    /// Click turns on the sensors for that flock.
    /// </summary>
    /// <param name="sender"></param>
    /// <param name="e"></param>
    private void PictureBox1_MouseDown(object? sender, MouseEventArgs e)
    {
        if (sender is null) return;

        // to work out the dots, I did this on a full size play area: File.AppendAllText(@"c:\temp\2.txt", $"{e.Location}\n");
        LearnToHerd.Monitor((int)((PictureBox) sender).Tag);
    }

    /// <summary>
    /// If in quiet mode whilst closing it won't close until it yields. By setting s_silentMode = false, it comes out the loop.
    /// One might think "concurrent update causes corruption?" It shouldn't for simple data types like bool (it doesn't move memory
    /// location like an object might).
    /// </summary>
    /// <param name="sender"></param>
    /// <param name="e"></param>
    private void Form1_FormClosing(object sender, FormClosingEventArgs e)
    {
        LearnToHerd.s_silentMode = false;
        e.Cancel = false;
    }
}