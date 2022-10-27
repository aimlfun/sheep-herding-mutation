//#define spinInCirclesForTesting
//#define DrawArrowShowingAngleFromDogToHerd
using SheepHerderMutation;
using SheepHerderMutation.AI;
using SheepHerderMutation.Configuration;
using SheepHerderMutation.Sheepies;
using SheepHerderMutation.Utilities;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace SheepHerderMutation.Predator;

/// <summary>
/// 
/// </summary>
internal class Dog
{

    //   ____              
    //  |  _ \  ___   __ _ 
    //  | | | |/ _ \ / _` |
    //  | |_| | (_) | (_| |
    //  |____/ \___/ \__, |
    //               |___/ 

    /// <summary>
    /// The link between dog brain and dog. 
    /// Brain's get preserved or mutated. Animals are spawned with existing brains.
    /// </summary>
    readonly private int Id;

    /// <summary>
    /// Which flock of sheep the dog is herding (given multiple dogs, and hence flocks of sheep).
    /// </summary>
    readonly Flock flockBeingHerded;

    /// <summary>
    /// Where the predator is located.
    /// </summary>
    internal PointF Position = new(10, 10);

    /// <summary>
    /// Where we want the dog to go (human mouse position, computed via heuristic or AI).
    /// </summary>
    internal PointF DesiredPosition = new(10, 10);

    /// <summary>
    /// The direction the predator is facing.
    /// </summary>
    internal float AngleDogIsFacingInDegrees = 0;

    /// <summary>
    /// How fast the predator is moving.
    /// </summary>
    internal float Speed = 0;

    /// <summary>
    /// Sets a flag to indicate dog is being monitored. When enabled, it draws the sensors.
    /// </summary>
    public bool MonitoringEnabled { get; internal set; }

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="id"></param>
    /// <param name="flockItbelongsTo"></param>
    internal Dog(int id, Flock flockItbelongsTo)
    {
        Id = id;
        flockBeingHerded = flockItbelongsTo;
    }

    /// <summary>
    /// Move the dog. 
    /// </summary>
    internal void Move()
    {
        List<double> inputToAI = InputsToAI();

        if (Config.AIDeriveOptimalPositionForDog)
        {
            SetDesiredPositionByUsingAItoSteerTheDog(inputToAI);
            SetAngleAndSpeedBasedOnDesiredPosition();
        }
        else
        {
            SetDesiredAngleAndSpeedUsingAItoSteerTheDog(inputToAI);
        }

        // Sin/Cos are in radians, so we need to convert
        float angleInRadians = (float)MathUtils.DegreesInRadians(AngleDogIsFacingInDegrees);

        // predator moves towards the chosen angle, at the chosen speed.
        Position.X += Speed * (float)Math.Cos(angleInRadians);
        Position.Y += Speed * (float)Math.Sin(angleInRadians);

        PreventDogJumpingFences();

        // ensure they don't go outside the UI area.
        Position.X = Position.X.Clamp(3, LearnToHerd.s_sizeOfPlayingField.Width - 3);
        Position.Y = Position.Y.Clamp(3, LearnToHerd.s_sizeOfPlayingField.Height - 3);
    }

    /// <summary>
    /// Instead of assigning a position for the dog, this tells it to rotate.
    /// </summary>
    /// <param name="inputToAI"></param>
    private void SetDesiredAngleAndSpeedUsingAItoSteerTheDog(List<double> inputToAI)
    {
        // ask the AI what to do next?
        double[] output = NeuralNetwork.s_networks[Id].FeedForward(inputToAI.ToArray());

        // output 0 = angle required; clamped for realistic behaviour to +/15 degrees rotation per frame
        AngleDogIsFacingInDegrees = MathUtils.Clamp360(AngleDogIsFacingInDegrees + (float)output[0]);

        // output 1 of neural network = the speed to travel.
        Speed = (float)output[1] * Config.DogSpeedMultiplier;

        if (Speed == 0) Speed = 0.5f;

        Speed = Speed.Clamp(-Config.DogMaximumVelocityInAnyDirection, Config.DogMaximumVelocityInAnyDirection);
    }

    /// <summary>
    /// Uses a neural network to move the dog.
    /// </summary>
    /// <param name="inputToAI"></param>
    private void SetDesiredPositionByUsingAItoSteerTheDog(List<double> inputToAI)
    {
        // ask the AI what to do next? inputs[] => feedforward => outputs[],
        // [0] = x offset from centreOfMass to move dog to [1] = y offset from centreOfMass to move dog to

        // the underlying calculation requires sin/cos, tanh, closest point on a line, so instead we train the AI
        // where the dog should go to steeer the flock in the intended direction. It has to use offsets otherwise
        // the training would be specific to a course and exact location of the sheep and dog.
        double[] output = NeuralNetwork.s_networks[Id].FeedForward(inputToAI.ToArray());

        PointF centreOfMass = flockBeingHerded.TrueCentreOfMass(out _);

        DesiredPosition.X = (float)(centreOfMass.X + output[0] * LearnToHerd.s_sizeOfPlayingField.Width);
        DesiredPosition.Y = (float)(centreOfMass.Y + output[1] * LearnToHerd.s_sizeOfPlayingField.Height);
    }

    /// <summary>
    /// All 3 control mechanisms set DesiredPosition (where dog needs to be).
    /// 
    /// We now need to work out how the dog should respond - which way to rotate, what speed to travel.
    /// </summary>
    private void SetAngleAndSpeedBasedOnDesiredPosition()
    {
        float angleInDegrees = (float)MathUtils.RadiansInDegrees((float)Math.Atan2(DesiredPosition.Y - Position.Y, DesiredPosition.X - Position.X));

        float deltaAngle = Math.Abs(angleInDegrees - AngleDogIsFacingInDegrees).Clamp(0, 30);

        // quickest way to get from current angle to new angle turning the optimal direction
        float angleInOptimalDirection = (angleInDegrees - AngleDogIsFacingInDegrees + 540f) % 360 - 180f;

        // limit max of 30 degrees
        AngleDogIsFacingInDegrees = MathUtils.Clamp360(AngleDogIsFacingInDegrees + deltaAngle * Math.Sign(angleInOptimalDirection));

        // close the distance as quickly as possible but without the dog going faster than it should
        Speed = MathUtils.DistanceBetweenTwoPoints(Position, DesiredPosition).Clamp(-Config.DogMaximumVelocityInAnyDirection, Config.DogMaximumVelocityInAnyDirection);
    }

    /// <summary>
    /// Inputs are 
    /// </summary>
    /// <returns></returns>
    private List<double> InputsToAI()
    {
#pragma warning disable CS0162 // Unreachable code detected - config decides if code executes. You can ignore this error.

        List<double> inputToAI = new();

        PointF centreOfMass = flockBeingHerded.TrueCentreOfMass(out _);

        //if (Config.AIknowsAngleOfDog) inputToAI.Add(MathUtils.DegreesInRadians(Angle / 360) / Math.PI);
        if (Config.AIknowsAngleOfDog) inputToAI.Add((MathUtils.DegreesInRadians(AngleDogIsFacingInDegrees) - Math.PI) / Math.PI); // 2*PI = 360, so +/-PI = -180..180

        double angle = Config.AIZeroRelativeAnglesInSensors ? 0 : AngleDogIsFacingInDegrees;

        // sense where the flock of sheep are located
        if (Config.AIUseSheepSensor)
        {
            SheepSensor sensor = new();
            sensor.Read(angle, Position, flockBeingHerded.flock, out double[] heatSensorRegionsOutput);

            inputToAI.AddRange(heatSensorRegionsOutput);
        }

        // flock is out of sight to predator
        if (!SheepWereSensedInObservableArea())
        {
            flockBeingHerded.failureReason = "OUT OF SIGHT";
            flockBeingHerded.flockIsFailure = true;
        }

        // inform the AI whether it is close to a wall or not.
        if (Config.AIusingWallSensors)
        {
            WallSensor wallSensor = new();
            wallSensor.Read(angle, Position, out double[] heatSensorRegionsOutput2);
            inputToAI.AddRange(heatSensorRegionsOutput2);
        }


        if (Config.AIknowsDistanceToSheepCenterOfMass) inputToAI.Add(MathUtils.DistanceBetweenTwoPoints(Position, centreOfMass) / Config.DogSensorOfSheepVisionDepthOfVisionInPixels);

        if (Config.AIKnowsAngleRelativeToHerd)
        {
            inputToAI.Add(Math.Atan2(centreOfMass.Y - Position.Y, centreOfMass.X - Position.X) / Math.PI);
        }

        // relative to...
        if (Config.AIknowsRelativeDistanceDogToCM)
        {
            // sheep dogs know where they are in the field, so we give that to the AI
            inputToAI.Add((centreOfMass.X - Position.X) / LearnToHerd.s_sizeOfPlayingField.Width);
            inputToAI.Add((centreOfMass.Y - Position.Y) / LearnToHerd.s_sizeOfPlayingField.Height);
        }

        if (Config.AIknowsWhereDogIsOnTheScreen)
        {
            // sheep dogs know where they are in the field, so we give that to the AI
            inputToAI.Add(Position.X / LearnToHerd.s_sizeOfPlayingField.Width);
            inputToAI.Add(Position.Y / LearnToHerd.s_sizeOfPlayingField.Height);
        }

        if (Config.AIknowsAngleSheepAreMoving) inputToAI.Add(flockBeingHerded.GetAngleFlockIsMovingInRadians() / Math.PI);

        if (Config.AIknowsPositionOfCenterOfMass)
        {
            inputToAI.Add(centreOfMass.X / LearnToHerd.s_sizeOfPlayingField.Width);
            inputToAI.Add(centreOfMass.Y / LearnToHerd.s_sizeOfPlayingField.Height);
        }

        // direction we want the sheep to "collectively" go
        if (Config.AIknowsAngleSheepNeedToHead) inputToAI.Add(flockBeingHerded.AngleToNextWayPoint() / Math.PI);

        if (Config.AIknowHowCloseItCanGetToSheep)
        {
            inputToAI.Add(ClosestDogMayIntentionallyGetToSheepMass() / LearnToHerd.s_sizeOfPlayingField.Width);
        }
#pragma warning restore CS0162 // Unreachable code detected

        return inputToAI;
    }

    /// <summary>
    /// Number of pixels dog must attempt to keep away from sheep centre of mass.
    /// </summary>
    /// <param name="pherd"></param>
    /// <returns></returns>
    internal static float ClosestDogMayIntentionallyGetToSheepMass()
    {
        // hard-code, because if we compute based on all the sheep, stragglers kill the algorithm
        return 57;
    }

    /// <summary>
    /// Look at all the outputs of sheep sensor to see if any sheep are present.
    /// </summary>
    /// <param name="heatSensorRegionsOutput"></param>
    /// <returns>true - sheep are within sensor</returns>
    private bool SheepWereSensedInObservableArea()
    {
        bool seeSheep = false;

        foreach (Sheep sheep in flockBeingHerded.flock)
        {
            if (MathUtils.DistanceBetweenTwoPoints(Position, sheep.Position) < Config.DogSensorOfSheepVisionDepthOfVisionInPixels)
            {
                seeSheep = true;
                break;
            }
        }

        return seeSheep;
    }

    /// <summary>
    /// Collision detect: We check to see if the dog's position is now within the lines 
    /// representing fences.
    /// </summary>
    private void PreventDogJumpingFences()
    {
        PointF c = new();

        // collision with any walls?
        foreach (PointF[] points in LearnToHerd.s_lines)
        {
            for (int i = 0; i < points.Length - 1; i++) // -1, because we're doing line "i" to "i+1"
            {
                PointF point1 = points[i];
                PointF point2 = points[i + 1];

                // touched wall? returns the closest point on the line to the sheep. We check the distance
                if (MathUtils.IsOnLine(point1, point2, new PointF(Position.X + c.X, Position.Y + c.Y), out PointF closest) &&
                    MathUtils.DistanceBetweenTwoPoints(closest, Position) < 6)
                {
                    // yes, need to back off from the wall
                    c.X -= (closest.X - Position.X) / 2;
                    c.Y -= (closest.Y - Position.Y) / 2;
                }
            }
        }

        Position.X += c.X;
        Position.Y += c.Y;
    }

    /// <summary>
    /// Draw the sheep in the sheep pen.
    /// 
    /// Initially we create as a filled in white circle, maybe later get a little more fancy.
    /// </summary>
    /// <param name="g"></param>
    internal void Draw(Graphics g)
    {
        // overlay a marker showing where we want the dog to go
        if (Config.AIDeriveOptimalPositionForDog) g.FillEllipse(Brushes.Yellow, DesiredPosition.X - 4, DesiredPosition.Y - 4, 8, 8);

        // show the predator as a filled black circle 
        g.FillEllipse(Brushes.Black /* new SolidBrush(flockBeingHunted.Black)*/, Position.X - 4, Position.Y - 4, 8, 8);

        DrawFaintCircleIndicatingDogsVisionLimit(g);

#if spinInCirclesForTesting
        Angle += 10; // (float)(Math.PI * 2) / 15;

        if (Angle > Math.PI * 2) Angle -= (float)Math.PI * 2;
        if (Angle >= 360) Angle -= 360;
#endif

        // if monitoring, we overlay the 2 sensors.
        if (MonitoringEnabled)
        {
            SheepSensor sensor = new();
            sensor.Read(AngleDogIsFacingInDegrees, Position, flockBeingHerded.flock, out double[] heatSensorRegionsOutput);

            sensor.DrawFullSweepOfHeatSensor(g, Color.FromArgb(20, 0, 255, 0));
            sensor.DrawWhereTargetIsInRespectToSweepOfHeatSensor(g, Color.FromArgb(30, 255, 0, 0));

#if spinInCirclesForTesting
            // show a line pointing where the predator is pointing
            PointF zz = new(
                Position.X + 50 * (float)Math.Cos(Utils.DegreesInRadians(Angle)),
                Position.Y + 50 * (float)Math.Sin(Utils.DegreesInRadians(Angle)));

            g.DrawLine(Pens.DarkBlue, Position, zz);
#endif

            WallSensor wallSensor = new();
            wallSensor.Read(AngleDogIsFacingInDegrees, Position, out double[] heatSensorRegionsOutput2);

            wallSensor.DrawFullSweepOfHeatSensor(g, Color.FromArgb(10, 0, 255, 0));
            wallSensor.DrawWhereTargetIsInRespectToSweepOfHeatSensor(g, new SolidBrush(Color.FromArgb(190, 255, 0, 0)));
        }

#if DrawArrowShowingAngleFromDogToHerd
        if (Config.AIKnowsAngleRelativeToHerd) DrawArrowShowingAngleFromDogToHerdIsCalculatedCorrectly(g);
#endif
    }

#if DrawArrowShowingAngleFromDogToHerd
    /// <summary>
    /// We provide the AI dog with an angle indicating the direction to the flock.
    /// This draws the arrow. It is fixed size unless the AI knows the distance to the flock.
    /// </summary>
    /// <param name="g"></param>
    private void DrawArrowShowingAngleFromDogToHerdIsCalculatedCorrectly(Graphics g)
    {
        // show an arrow indicating the angle it knows

        PointF pherd = flockBeingHerded.TrueCentreOfMass();
        double herdAngle = Math.Atan2(pherd.Y - Position.Y,
                                      pherd.X - Position.X);

        float size = 20;

        using Pen p2 = new(Color.Aqua);
        p2.DashStyle = DashStyle.Dot;
        p2.EndCap = LineCap.ArrowAnchor;

        // size the line to indicate it knows the distance
        if (Config.AIknowsDistanceToSheepCenterOfMass)
        {
            PointF p = flockBeingHerded.TrueCentreOfMass();
            size = (float)MathUtils.DistanceBetweenTwoPoints(Position, pherd);
        }

        g.DrawLine(p2,
                   (int)Position.X, (int)Position.Y,
                   (int)(size * Math.Cos(herdAngle) + Position.X),
                   (int)(size * Math.Sin(herdAngle) + Position.Y));
    }
#endif

    /// <summary>
    /// Having a circle helps you realise when the AI moves to far away from the 
    /// center of mass.
    /// </summary>
    /// <param name="g"></param>
    private void DrawFaintCircleIndicatingDogsVisionLimit(Graphics g)
    {
        // draw a faint circle showing the diameter of the dog's vision

        using Pen p = new(Color.FromArgb(30, 255, 255, 255));
        p.DashStyle = DashStyle.Dot;
        g.DrawEllipse(
            p,
            (int)(Position.X - Config.DogSensorOfSheepVisionDepthOfVisionInPixels),
            (int)(Position.Y - Config.DogSensorOfSheepVisionDepthOfVisionInPixels),
            (int)Config.DogSensorOfSheepVisionDepthOfVisionInPixels * 2,
            (int)Config.DogSensorOfSheepVisionDepthOfVisionInPixels * 2);
    }
}