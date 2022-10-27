#define showNumberingOfWayPoints
using Microsoft.VisualBasic.Devices;
using SheepHerderMutation.AI;
using SheepHerderMutation.Configuration;
using SheepHerderMutation.Sheepies;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace SheepHerderMutation;

///    _                        _____     _   _              _ 
///   | |    ___  __ _ _ __ _ _|_   _|__ | | | | ___ _ __ __| |
///   | |   / _ \/ _` | '__| '_ \| |/ _ \| |_| |/ _ \ '__/ _` |
///   | |__|  __/ (_| | |  | | | | | (_) |  _  |  __/ | | (_| |
///   |_____\___|\__,_|_|  |_| |_|_|\___/|_| |_|\___|_|  \__,_|
///
static class LearnToHerd
{
    /// <summary>
    /// Delegate for event handler (we call upon mutation).
    /// </summary>
    public delegate void MutationDelegate();

    /// <summary>
    /// When mutating event. We hook into this to provide graphs.
    /// </summary>
    internal static event MutationDelegate? WhenMutating;

    /// <summary>
    /// This is the green field with brown hedges/fences, and a score zone. 
    /// We paint it once, and use it as the basis of our background.
    /// </summary>
    internal static Bitmap? s_backgroundImage;

    /// <summary>
    /// Size of the "learning" area (sheep pen / fences).
    /// </summary>
    internal static Size s_sizeOfPlayingField = new();

    /// <summary>
    /// Brush for drawing scoring zone. Static as it applies to ALL flocks.
    /// </summary>
    private readonly static HatchBrush s_hatchBrushForScoringZone = new(HatchStyle.DiagonalCross, Color.FromArgb(30, 255, 255, 255), Color.Transparent);

    /// <summary>
    /// Region that is the "home" scoring zone. Static as it applies to ALL flocks.
    /// </summary>
    internal static RectangleF s_sheepPenScoringZone = new(100, 100, 100, 100);

    /// <summary>
    /// when true, we don't draw the sheep (runs quicker).
    /// </summary>
    internal static bool s_silentMode = false;

    /// <summary>
    /// Lines the sheep must avoid (makes for more of a challenge). Static as it applies to ALL flocks.
    /// </summary>
    internal static List<PointF[]> s_lines = new();

    /// <summary>
    /// The generation (how many times the network has been mutated).
    /// </summary>
    internal static float s_generation = 0;

    /// <summary>
    /// The list of flocks indexed by their "id".
    /// </summary>
    internal readonly static Dictionary<int, Flock> s_flock = new();

    /// <summary>
    /// If set to true, then this ignores requests to mutate.
    /// </summary>
    internal static bool s_stopMutation = false;

    /// <summary>
    /// How many moves the predator has performed.
    /// </summary>
    internal static int s_numberOfMovesMadeByPredator = 0;

    /// <summary>
    /// Defines the number of moves it will initialise the mutate counter to.
    /// This number increases with each generation.
    /// </summary>
    internal static int s_movesToCountBetweenEachMutation = 0;

    /// <summary>
    /// Defines the number of moves before a mutation occurs. 
    /// This is decremented each time the cars move, and upon reaching zero triggers
    /// a mutation.
    /// </summary>
    internal static int s_movesLeftBeforeNextMutation = 0;

    /// <summary>
    /// The points the sheep need to go thru.
    /// </summary>
    internal static Point[] s_wayPointsSheepNeedsToGoThru = Array.Empty<Point>();

    /// <summary>
    /// Used for drawing the scoreboard. 
    /// </summary>
    private readonly static Font scoreBoardFont = new("Arial", 8);

    /// <summary>
    /// Start the AI learning process: initialises the neural networks, and starts first generation.
    /// </summary>
    internal static void StartLearning(int width, int height)
    {
        s_sizeOfPlayingField = new Size(width, height);

        s_backgroundImage ??= DrawbackgroundImage();

        InitialiseNeuralNetworkLayersRequiredForSettings();

        s_generation = 0;

        InitialiseTheNeuralNetworksForEachFlock();
        s_movesToCountBetweenEachMutation = Config.AINumberOfInitialMovesBeforeFirstMutation;

        NextGeneration();
    }

    /// <summary>
    /// As config allows inputs to be enabled/disabled, it's necessary to adjust the layer counts.
    /// </summary>
    private static void InitialiseNeuralNetworkLayersRequiredForSettings()
    {
        int inputNeuronsCount = DetermineRequiredNumberOfInputNeurons();

        Config.AIHiddenLayers[0] = inputNeuronsCount;
        Config.AIHiddenLayers[^1] = 2; // we have speed and direction

        for (int i = 1; i < Config.AIHiddenLayers.Length - 1; i++)
        {
            if (Config.AIHiddenLayers[i] == 0) Config.AIHiddenLayers[i] = inputNeuronsCount;
            if (Config.AIHiddenLayers[i] < 0) throw new Exception("The number of neurons in a neural networks cannot be negative. What would that even mean?");
        }
    }

    /// <summary>
    /// Inputs to neural network is configurable, requiring us to determine the neuron count
    /// required based on what is enabled/disabled.
    /// </summary>
    /// <returns></returns>
    private static int DetermineRequiredNumberOfInputNeurons()
    {
        int sheepSensorOutputCount = Config.AIUseSheepSensor ? (int)(360F / Config.DogSensorOfSheepVisionAngleInDegrees) : 0;
        int wallSensorSamplePoints = Config.AIusingWallSensors ? Config.DogWallSensorSamplePoints : 0; int locationKnown = Config.AIknowsWhereDogIsOnTheScreen ? 2 : 0;
        int knowsSheepCoM = Config.AIknowsDistanceToSheepCenterOfMass ? 1 : 0;
        int sheepDirector = Config.AIknowsAngleSheepNeedToHead ? 1 : 0;
        int comPosition = Config.AIknowsRelativeDistanceDogToCM ? 2 : 0;
        int angleOfDog = Config.AIknowsAngleOfDog ? 1 : 0;
        int angleBetweenDogAndHerd = Config.AIKnowsAngleRelativeToHerd ? 1 : 0;
        int angleFlockIsMoving = Config.AIknowsAngleSheepAreMoving ? 1 : 0;
        int comPosAbsolute = Config.AIknowsPositionOfCenterOfMass ? 2 : 0;

        return sheepSensorOutputCount + wallSensorSamplePoints + locationKnown + knowsSheepCoM + sheepDirector + comPosition + angleOfDog + angleBetweenDogAndHerd + angleFlockIsMoving + comPosAbsolute;
    }

    /// <summary>
    /// Moves to next generation. Increases the moves by a fixed %age.
    /// Flock is "initialised".
    /// </summary>
    internal static void NextGeneration()
    {
        if (s_movesLeftBeforeNextMutation == 0) s_movesLeftBeforeNextMutation = s_movesToCountBetweenEachMutation;

        // enable each mutation to have longer to run, so the cars go further.
        s_movesToCountBetweenEachMutation = s_movesLeftBeforeNextMutation * (100 + Config.AIPercentageIncreaseBetweenMutations) / 100;

        s_movesLeftBeforeNextMutation = s_movesToCountBetweenEachMutation;

        // after a mutation, we crush cars and get new ones (reset their position/state) whilst keeping the neural networks
        InitialiseFlocks();

        s_numberOfMovesMadeByPredator = 0;
    }

    /// <summary>
    /// Initialises the neural network for the sheep.
    /// </summary>
    internal static void InitialiseTheNeuralNetworksForEachFlock()
    {
        NeuralNetwork.s_networks.Clear();

        for (int i = 0; i < Config.NumberOfAIdogs; i++)
        {
            _ = new NeuralNetwork(i, Config.AIHiddenLayers, Config.AIactivationFunctions);
        }

        bool loaded = false;

        foreach (NeuralNetwork n in NeuralNetwork.s_networks.Values)
        {
            //loaded |= n.Load(@"c:\temp\sheep0-UHA.ai");

            loaded |= n.Load($@"c:\temp\sheep{n.Id}.ai");
        }

        // loaded are trained. Trained don't need an early mutation.
        if (loaded) Config.AINumberOfInitialMovesBeforeFirstMutation = 100000;
    }

    /// <summary>
    /// If not the first flock, kills them humanely first. 
    /// Re-creates the flock (reset position etc). Important: this does not touch the neural network we are training
    /// </summary>
    internal static void InitialiseFlocks()
    {
        s_generation++;

        // if we have flocks, then we need to mutate the brains based of fitness (a brain per dog)
        if (s_flock.Count > 0)
        {
            MutateFlock();
            s_flock.Clear();
        }

        List<PointF> sheepPositions = new();
        for (int i = 0; i < Config.InitialFlockSize; i++)
        {
            // randomish place near the start position
            sheepPositions.Add(new PointF(RandomNumberGenerator.GetInt32(0, s_sizeOfPlayingField.Width / 6 + 20),
                                          RandomNumberGenerator.GetInt32(0, s_sizeOfPlayingField.Height / 6) + 40));
        }


        // we train multiple dogs at once, this creates a dog and its flock
        for (int flockNumber = 0; flockNumber < Config.NumberOfAIdogs; flockNumber++)
        {
            Flock flock = new(flockNumber, sheepPositions, s_sizeOfPlayingField.Width, s_sizeOfPlayingField.Height);

            s_flock.Add(flockNumber, flock);
        }
    }

    /// <summary>
    /// If you click the picturebox, this monitors the dog and flock.
    /// </summary>
    /// <param name="p"></param>
    internal static void Monitor(int id)
    {
        s_flock[id].dog.MonitoringEnabled = !s_flock[id].dog.MonitoringEnabled;
    }

    /// <summary>
    /// Mutation of the neural network.
    /// </summary>
    private static void MutateFlock()
    {
        if (Config.NumberOfAIdogs == 1)
        {
            NeuralNetwork.s_networks[0] = new NeuralNetwork(0, Config.AIHiddenLayers, Config.AIactivationFunctions, false);
            return;
        }

        float max = -1;

        float maxSuccess = 0;

        // update networks fitness for each car
        foreach (int id in s_flock.Keys)
        {
            NeuralNetwork.s_networks[id].Mutated = false;
            NeuralNetwork.s_networks[id].GenerationOfLastMutation++;
            NeuralNetwork.s_networks[id].Fitness = s_flock[id].FitnessScore();

            float thisSuccess = NeuralNetwork.s_networks[id].BestFitness();
            if (thisSuccess > maxSuccess) maxSuccess = thisSuccess;

            NeuralNetwork.s_networks[id].Performance.Add((int)NeuralNetwork.s_networks[id].Fitness);

            if (NeuralNetwork.s_networks[id].Fitness > max) max = NeuralNetwork.s_networks[id].Fitness;
        }

        NeuralNetwork.SortNetworkByFitness(); // largest "fitness" (best performing) goes to the bottom

        // sorting is great but index no longer matches the "id".
        // this is because the sort swaps but this misaligns id with the entry            
        List<NeuralNetwork> n = new();
        foreach (int n2 in NeuralNetwork.s_networks.Keys) n.Add(NeuralNetwork.s_networks[n2]);

        NeuralNetwork[] array = n.ToArray();

        // replace the 50% worse offenders with the best, then mutate them.
        // we do this by copying top half (lowest fitness) with top half.
        for (int worstNeuralNetworkIndex = 0; worstNeuralNetworkIndex < Config.NumberOfAIdogs / 2; worstNeuralNetworkIndex++)
        {
            // if one has X successes, then new generations need the chance to get a similar number before we write them off
            if (array[worstNeuralNetworkIndex].GenerationOfLastMutation >= maxSuccess)
            {
                // 50..100 (in 100 neural networks) are in the top performing
                int neuralNetworkToCloneFromIndex = worstNeuralNetworkIndex + Config.NumberOfAIdogs / 2; // +50% -> top 50% 

                //int neuralNetworkToCloneFromIndex = Config.NumberOfAIdogs -1; // best 

                if (array[neuralNetworkToCloneFromIndex].GenerationOfLastMutation < maxSuccess) return; // don't clone new ones, as performance isn't proven

                NeuralNetwork.CopyFromTo(array[neuralNetworkToCloneFromIndex], array[worstNeuralNetworkIndex]); // copy
                array[worstNeuralNetworkIndex].GenerationOfLastMutation = 0;

                array[worstNeuralNetworkIndex].Mutate(5, 0.25F); // mutate
                array[worstNeuralNetworkIndex].Performance.Clear(); // start clean slate
                array[worstNeuralNetworkIndex].Performance.Add(Math.Min(26, (int)array[neuralNetworkToCloneFromIndex].AverageFitness() - 1));
            }
        }

        // randomise replacement for the worst
        NeuralNetwork.s_networks[array[0].Id] = new NeuralNetwork(array[0].Id, Config.AIHiddenLayers, Config.AIactivationFunctions, false)
        {
            Fitness = array[(int)(Config.NumberOfAIdogs * 0.75f)].AverageFitness(),  // give it an average fitness so it doesn't get instantly excluded
            GenerationOfLastMutation = 0
        };

        // unsort, restoring the order of car to neural network i.e [x]=id of "x".
        Dictionary<int, NeuralNetwork> unsortedNetworksDictionary = new();

        for (int predatorIndex = 0; predatorIndex < Config.NumberOfAIdogs; predatorIndex++)
        {
            var neuralNetwork = NeuralNetwork.s_networks[predatorIndex];

            unsortedNetworksDictionary[neuralNetwork.Id] = neuralNetwork;
        }

        NeuralNetwork.s_networks = unsortedNetworksDictionary;
    }

    /// <summary>
    /// Move all the sheep and predators (occurs when the timer fires)
    /// </summary>
    internal static void Learn()
    {
        // in silent mode, we run in a loop painting nothing, not waiting for next timer tick.
        // that runs very fast
        if (s_silentMode)
        {
            Form1.s_timer.Stop(); // we're doing it via a loop.
            Form1.s_timer.Interval = 1;
        }

        bool loop = true; // true;

        while (loop)
        {
            // interval 1 means move in a loop, yielding occasionally, it's quicker than waiting 1ms for the timer.
            if (!s_silentMode) loop = false;

            ++s_numberOfMovesMadeByPredator;

            if (s_numberOfMovesMadeByPredator > s_movesLeftBeforeNextMutation)
            {
                NextGeneration();

                if (s_silentMode) WhenMutating?.Invoke();
            }

            MoveAllFlocks();

            // if we don't do this, it ends hanging the UI. This provides enough Windows message loop
            if (s_silentMode && s_numberOfMovesMadeByPredator % 200 == 0) Application.DoEvents();
        }

        // restart timer
        if (Form1.s_timer.Interval == 1)
        {
            Form1.s_timer.Interval = 10;
            Form1.s_timer.Start();
        }
    }

    /// <summary>
    /// Moves all the flocks at once, in parallel or serial.
    /// Use serial for debugging, and parallel to get very fast performance.
    /// </summary>
    private static void MoveAllFlocks()
    {
        if (Config.SheepUseParallelComputation)
        {
            // all sheep are independent, each one has a neural network and sensors attached, this therefore
            // is a candidate for parallelism.
            Parallel.ForEach(s_flock.Keys, id =>
            {
                s_flock[id].Move();
            });
        }
        else
        {
            foreach (int id in s_flock.Keys)
            {
                s_flock[id].Move();
            }
        }
    }

    /// <summary>
    /// Draws ALL the dogs along with their respective flock.
    /// </summary>
    /// <returns></returns>
    internal static List<Bitmap> DrawAll(out bool mutateNow)
    {
        mutateNow = false;

        int failures = 0;

        List<Bitmap> images = new();

        foreach (int id in s_flock.Keys)
        {
            if (s_flock[id].flockIsFailure) ++failures;
        }

        // if all are failures, we need to mutate.
        mutateNow = failures == s_flock.Keys.Count;

        // better performance for AI learning if we DON'T draw everything
        if (s_silentMode) return images; // 0 images

        if (Config.SheepUseParallelComputation)
        {
            ConcurrentDictionary<int, Bitmap> imageDictionary = new();

            Parallel.ForEach(s_flock.Keys, id =>
            {
                imageDictionary.TryAdd(id, DrawFlockToImage(s_flock[id]));
            });

            images = imageDictionary.Values.ToList();
        }
        else
        {
            foreach (int id in s_flock.Keys)
            {
                Flock flock = s_flock[id];
                images.Add(DrawFlockToImage(flock));
            }
        }

        return images;
    }

    /// <summary>
    /// Draws the flock to a bitmap.
    /// </summary>
    /// <param name="flock"></param>
    /// <returns></returns>
    private static Bitmap DrawFlockToImage(Flock flock)
    {
        Bitmap image;

        // we don't waste drawing it, if the flock is a failure; nothing moves.
        if (flock.flockIsFailure && flock.cachedImageAfterDeath is not null)
            image = new Bitmap(flock.cachedImageAfterDeath);
        else
        {
            // each flock is a separate image, that starts with a predefined background
            image = new(flock.s_backgroundImage, s_sizeOfPlayingField);
            using Graphics graphics = Graphics.FromImage(image);
            graphics.CompositingQuality = CompositingQuality.HighQuality;
            graphics.SmoothingMode = SmoothingMode.HighQuality;

            flock.Draw(graphics);

            // add a "layer" that darkens the image to indicate failure and make it clear which ones are finished, and still running
            if (flock.flockIsFailure)
            {
                using SolidBrush p = new(Color.FromArgb(150, 0, 0, 0));
                graphics.FillRectangle(p, new Rectangle(0, 0, s_sizeOfPlayingField.Width, s_sizeOfPlayingField.Height));
            }

            float prevFitness = NeuralNetwork.s_networks[flock.Id].Fitness;
            graphics.DrawString((prevFitness > Flock.c_exitQuietModeWhenScoreReached ? "** " : "") + $"Id: {flock.Id}  score {flock.numberOfSheepInPenZone}   fitness {Math.Round(flock.FitnessScore())} ({prevFitness})   {flock.failureReason}",
                                scoreBoardFont, Brushes.White, 10, 10);

            graphics.Flush();

            // if we got here due to failure, but image is null, we store the image to avoid painting next time.
            if (flock.flockIsFailure)
                flock.cachedImageAfterDeath = new Bitmap(image);
        }

        return image;
    }

    /// <summary>
    /// Paints the background that is common to all flocks (field, fences, score zone)
    /// </summary>
    /// <returns></returns>
    internal static Bitmap DrawbackgroundImage()
    {
        Bitmap image = new(s_sizeOfPlayingField.Width, s_sizeOfPlayingField.Height);

        using Graphics g = Graphics.FromImage(image);
        g.Clear(Color.Green);
        g.CompositingQuality = CompositingQuality.HighQuality;
        g.SmoothingMode = SmoothingMode.HighQuality;

        // draw the scoring zone as a hatched area
        g.FillRectangle(s_hatchBrushForScoringZone, s_sheepPenScoringZone);

        // draw the lines around the play area (fences)
        using Pen p = new(Color.Brown, 4);
        foreach (PointF[] points in s_lines) g.DrawLines(p, points);

        // draw blobs for way points
        SolidBrush wayPointBrush = new(Color.FromArgb(30, 220, 230, 243));

#if showNumberingOfWayPoints
        int i = 1;

        foreach (Point wp in s_wayPointsSheepNeedsToGoThru)
        {
            g.FillEllipse(wayPointBrush, wp.X - 3, wp.Y - 3, 6, 6);
            g.DrawString(i.ToString(), new Font("Arial", 7), wayPointBrush, wp.X + 3, wp.Y);
            ++i;
        }

#else
        foreach (Point wp in LearnToHerd.s_wayPointsSheepNeedsToGoThru)
        {
            g.FillEllipse(wayPointBrush, wp.X - 3, wp.Y - 3, 6, 6);
        }
#endif


        g.Flush();

        return image;
    }
}