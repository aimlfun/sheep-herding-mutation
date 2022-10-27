using SheepHerderMutation.AI;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace SheepHerderMutation.Configuration;

/// <summary>
///    ____             __ _                       _   _             
///   / ___|___ _ __   / _(_) __ _ _   _ _ __ __ _| |_(_) ___ _ __  
///  | |   / _ \| '_ \| |_| |/ _` | | | | '__/ _` | __| |/ _ \| '_ \ 
///  | |__| (_) | | | |  _| | (_| | |_| | | | (_| | |_| | (_) | | | |
///   \____\___/|_| |_|_| |_|\__, |\__,_|_|  \__,_|\__|_|\___/|_| |_|
///                          |___/                                   
/// </summary>
internal static class Config
{
    //  ____  _                     
    // / ___|| |__   ___  ___ _ __
    // \___ \| '_ \ / _ \/ _ \ '_ \ 
    //  ___) | | | |  __/  __/ |_) |
    // |____/|_| |_|\___|\___| .__/ 
    //                       |_|    

    /// <summary>
    /// Set this to move the sheep using parallel processing (faster).
    /// </summary>
    internal static bool SheepUseParallelComputation = true;

    /// <summary>
    /// Defines how many sheep we create in a flock for the herding.
    /// </summary>
    internal static int InitialFlockSize = 20;

    /// <summary>
    /// The smaller the number, the closer the dog can get to the sheep before they react.
    /// </summary>
    internal static int SheepHowFarAwayItSpotsTheDog = 140; //90

    /// <summary>
    /// Maximum distance the sheep can be from any way-point. We do this optimise
    /// finding the right path.
    /// </summary>
    internal static double SheepMaxDeviationFromWayPoint = 50f;

    /// <summary>
    /// 
    /// </summary>
    internal static double SheepClosenessToMoveToNextWayPoint = 35f;

    /// <summary>
    /// If the sheep is slower than this, we make it stop.
    /// </summary>
    internal static float SheepMinimumSpeedBeforeStop = 0.1f;

    /// <summary>
    /// Sheep can run, but like all animals they are limited by physics and physiology.
    /// This controls the amount sheep can move per frame. It assumes each sheep is 
    /// comparable in performance.
    /// An average is 25mph for a sheep.
    /// </summary>
    internal static float SheepMaximumVelocityInAnyDirection = 0.7f;

    /// <summary>
    /// How close a sheep can sense all the other sheep (makes them clump).
    /// </summary>
    internal static float SheepCloseEnoughToBeAMass = 50;

    // MULTIPLIERS

    /// <summary>
    /// How much strength we apply to cohesion.
    /// </summary>
    internal const float SheepMultiplierCohesion = 0.5f;

    /// <summary>
    /// Use -ve to reflect how cohesion breaks down when a dog is herding.
    /// </summary>
    internal const float SheepMultiplierCohesionThreatenedByDog = -0.7f;

    /// <summary>
    /// How much strength we apply to keep the sheep separated.
    /// </summary>
    internal const float SheepMultiplierSeparation = 0.3f;

    /// <summary>
    /// Use -ve to reflect how separation is increased when a dog is herding.
    /// </summary>
    internal const float SheepMultiplierSeparationThreatenedByDog = -0.9f;

    /// <summary>
    /// How much alignement of movement.
    /// </summary>
    internal const float SheepMultiplierAlignment = 0.1f;

    /// <summary>
    /// Use -ve to reflect how much alignment is messed up when a dog is herding.
    /// </summary>
    internal const float SheepMultiplierAlignmentThreatenedByDog = 0.4f; //-1.9f;

    /// <summary>
    /// How much guidance is listened to. During herding, this is 0.
    /// </summary>
    internal const float SheepMultiplierGuidance = 0f;

    /// <summary>
    /// Use -ve to reflect how guidance is disrupted when a dog is herding.
    /// </summary>
    internal const float SheepMultiplierGuidanceThreatenedByDog = 0f;

    /// <summary>
    /// How much the presence of a predator (dog) impacts the escapel.
    /// </summary>
    internal const float SheepMultiplierEscape = 3f;


    //   ____              
    //  |  _ \  ___   __ _ 
    //  | | | |/ _ \ / _` |
    //  | |_| | (_) | (_| |
    //  |____/ \___/ \__, |
    //               |___/ 

    /// <summary>
    /// Defines how many dogs we run concurrently (with their own flocks of sheep).
    /// </summary>
    internal static int NumberOfAIdogs = 50; // must be multiple of 2

    /// <summary>
    /// If the dog takes more than this to progress movement of sheep thru a way point,
    /// it needs to be marked "failure". This provides better learning throughput.
    /// </summary>
    internal const int DogMaximumTimeWastingAllowedInSeconds = 50;

    /// <summary>
    /// How far from the center of the predator it looks for a wall.
    /// </summary>
    internal static double DogSensorWallDepthOfVisionInPixels = 10F;

    /// <summary>
    /// The number of points it checks to find walls.
    /// </summary>
    internal static int DogWallSensorSamplePoints = 8;

    /// <summary>
    /// How far away the dog sees the sheep.
    /// </summary>
    internal static double DogSensorOfSheepVisionDepthOfVisionInPixels = 140F; //90

    /// <summary>
    /// Defines how many "segments" the sensor has (365/this)
    /// </summary>
    internal static double DogSensorOfSheepVisionAngleInDegrees = 5.703125f;

    /// <summary>
    /// This boosts the AI output to make the dog move faster
    /// </summary>
    internal static float DogSpeedMultiplier = 4f;

    /// <summary>
    /// Dogs can run, but like all animals they are limited by physics and physiology.
    /// This controls the amount the dog can move per frame. It assumes each dog is 
    /// comparable in performance.
    /// An average is 30mph for a Collie sheep dog.
    /// </summary>
    internal static float DogMaximumVelocityInAnyDirection = 1.3f;

    //     _      ___   
    //    / \    |_ _|  
    //   / _ \    | |   
    //  / ___ \ _ | | _ 
    // /_/   \_(_)___(_)

    /// <summary>
    /// Defines the layers of the perceptron network. Each value is the number of neurons in the layer.
    /// 
    /// [0] is overridden, as it must match input data
    /// [^1] is overridden, as it is 2 (speed, direction)
    /// 
    /// Value of 0 => override with # of input/output neurons.
    /// </summary>
    internal static int[] AIHiddenLayers = { 0, 5, 10, 0 };

    /// <summary>
    /// This defines what activation functions to use. ONE per layer.
    /// </summary>
    internal static ActivationFunctions[] AIactivationFunctions = { ActivationFunctions.TanH,
                                                                    ActivationFunctions.TanH,
                                                                    ActivationFunctions.TanH,
                                                                    ActivationFunctions.Identity};

    /// <summary>
    /// The problem to solve becomes more complex if one rotates the sensor (0 being direction headed).
    /// true - 0 degrees indicates a specific zone (right), rotating clockwise.
    /// false - 0 degrees is the direction the dog is pointing.
    /// </summary>
    internal const bool AIZeroRelativeAnglesInSensors = true;

    /// <summary>
    /// true - it includes the wall sensor as a neural network input
    /// </summary>
    internal const bool AIusingWallSensors = false;

    /// <summary>
    /// true - sheep sensor number in the quadrant is 1 if sheep are present.
    /// false - number of sheep in quadrant / total sheep or if SheepSensorOutputIsDistance==true, then the distance 0..1.
    /// </summary>
    internal const bool AIBinarySheepSensor = false;

    /// <summary>
    /// true - the AI is given the angle of the dog relative to the herd.
    /// </summary>
    internal const bool AIKnowsAngleRelativeToHerd = false;

    /// <summary>
    /// true - the  sheep sensor is used (either binary or distance).
    /// </summary>
    internal const bool AIUseSheepSensor = true; // false

    /// <summary>
    /// true - the AI knows its location.
    /// false - the AI reacts to sheep and direction, without knowing where it is.
    /// </summary>
    internal const bool AIknowsWhereDogIsOnTheScreen = false;

    /// <summary>
    /// true - the AI knows its location.
    /// false - the AI reacts to sheep and direction, without knowing where it is.
    /// </summary>
    internal const bool AIknowsRelativeDistanceDogToCM = false; // true 

    /// <summary>
    /// true - the AI knows which way the dog is pointing.
    /// </summary>
    internal const bool AIknowsAngleOfDog = false;

    /// <summary>
    /// true - the dog knows how close it is to the sheep (without which it may fail "out of sight").
    /// false - the dog reacts to sensors with no real idea if it is close to sheep or not (closeness impacts sheep spread).
    /// 
    /// You would think having no awareness of sheep location in this manner shouldn't prevent it from achieving the goal,
    /// as the "sheep sensor" informs it that it is in range of sheep; however, the sensor when returning the "amount" of sheep
    /// or a binary presence doesn't enable it to know how close. You can compensate by setting "SheepSensorOutputIsDistance"=true
    /// </summary>
    internal const bool AIknowsDistanceToSheepCenterOfMass = false;

    /// <summary>
    /// true - the AI knows the centre of mass location for the sheep.
    /// </summary>
    internal const bool AIknowsPositionOfCenterOfMass = false;

    /// <summary>
    /// If true, the sheep sensor returns a number 0..1 that is proportional to the distance between dog and sheep.
    /// (scaled based on depth of vision).
    /// </summary>
    internal const bool AISheepSensorOutputIsDistance = false;

    /// <summary>
    /// true - the AI is told the angle the sheep need to go. This is equivalent to the dog learning the path.
    /// false - the AI has no idea what it is attempting, so don't expect it to guide sheep anywhere.
    /// </summary>
    internal const bool AIknowsAngleSheepNeedToHead = true; // true

    /// <summary>
    /// true - an extra parameter is given to the AI containing the angle the sheep are moving.
    /// </summary>
    internal const bool AIknowsAngleSheepAreMoving = false;

    /// <summary>
    /// After this amount of MOVES has elapsed, a mutation occurs.
    /// </summary>
    internal static int AINumberOfInitialMovesBeforeFirstMutation { get; set; } = 300; // moves

    /// <summary>
    /// This enables the learning to increase the time so that dogs get longer after
    /// each mutation. The idea being that they reach further, and weed out those that
    /// will eventually fail.
    /// </summary>
    internal static int AIPercentageIncreaseBetweenMutations { get; set; } = 5; // 0..100%

    /// <summary>
    /// If true, the AI knows how close to sheep it should get.
    /// </summary>
    public static bool AIknowHowCloseItCanGetToSheep { get; internal set; } = false;

    /// <summary>
    /// If true, it will decide where it wants to put the dog (x,y) cartesian coordinates.
    /// If false, it will rotate the dog and decide speed.
    /// </summary>
    public static bool AIDeriveOptimalPositionForDog { get; internal set; } = true;
}