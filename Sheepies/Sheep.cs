//#define drawVelocityArrows // <- draws arrows based on direction of velocity, and size
//#define drawCircleAroundCenterOfMass // <- draws a circle around the center of mass
using SheepHerderMutation.Configuration;
using SheepHerderMutation.Utilities;
using System;
using System.Collections.Generic;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace SheepHerderMutation.Sheepies;

/// <summary>
/// Represents a sheep (or white blob with dot for head).
/// The sheep moves with "flock" characteristics, and needs to be part of a flock.
/// </summary>
internal class Sheep
{

    //  ____  _                     
    // / ___|| |__   ___  ___ _ __
    // \___ \| '_ \ / _ \/ _ \ '_ \ 
    //  ___) | | | |  __/  __/ |_) |
    // |____/|_| |_|\___|\___| .__/ 
    //                       |_|    

    /// <summary>
    /// Solitary sheep are not as happy, so we group them together in a "flock".
    /// </summary>
    private readonly Flock flockSheepIsPartOf;

    /// <summary>
    /// Where the sheep is relative to the sheep pen.
    /// </summary>
    internal PointF Position = new();

    /// <summary>
    /// Preserved to replay - we need to know where this sheep started.
    /// </summary>
    internal PointF StartPosition = new();

    /// <summary>
    /// How fast the sheep is travelling (as a 2d vector) within sheep pen.
    /// </summary>
    internal PointF Velocity = new();

    /// <summary>
    /// The angle the sheep is pointing.
    /// </summary>
    internal float Angle = 0;

    /// <summary>
    /// Even sheep need to eat, if they are then this is set to true.
    /// </summary>
    internal bool IsStopped;

    /// <summary>
    /// Whilst > 0, sheep is paused (such as eating).
    /// </summary>
    internal int DurationRemainingStopped;

    /// <summary>
    /// Constructor. Sorry no lamb-das allowed.
    /// </summary>
    /// <param name="flock"></param>
    internal Sheep(Flock flock, PointF start)
    {
        if (flock is null) throw new ArgumentNullException(nameof(flock), "cannot be null, the sheep are designed to move in a flock");

        flockSheepIsPartOf = flock;

        // randomish place near the start position
        StartPosition = new PointF(start.X, start.Y);

        ResetToStart();
    }

    /// <summary>
    /// Used to initialise, and to put the sheep at the start.
    /// </summary>
    internal void ResetToStart()
    {
        Position = new PointF(StartPosition.X, StartPosition.Y);

        // stationary, but not for long...
        Velocity = new PointF(0, 0);

        // not stopped.
        IsStopped = false;

        Angle = 0;

        IsStopped = false;
    }

    /// <summary>
    /// Use to "stop" the sheep for a defined number of frames.
    /// </summary>
    /// <param name="frames"></param>
    internal void MakeSheepPause(int frames = 0)
    {
        if (frames < 1) frames = RandomNumberGenerator.GetInt32(1, 30); // pick a random amount of "pause" frames/time

        DurationRemainingStopped = frames;

        IsStopped = true;
    }

    /// <summary>
    /// The variation in strength of the second multiplier is described by a sigmoid function. 
    /// </summary>
    /// <param name="r">is the value of x where the absolute derivate of p(x) is the largest. This distance represents the flight zone radius of the sheep.</param>
    /// <param name="x">is the distance from the sheep to the dog.</param>
    /// <returns></returns>
    private static double DogDistanceSensitiveMultiplier(float r, float x)
    {
        return 1 / Math.PI * Math.Atan((r - x) / 20) + 0.5f;
    }

    /// <summary>
    /// Moves the sheep using flocking/swarming logic.
    /// </summary>
    internal void Move()
    {
        // sheep are grazing or something?
        if (IsStopped)
        {
            if (--DurationRemainingStopped > 0) return;

            IsStopped = false;
        }

        // compute distance sheep is from the dog
        float distToPredator = MathUtils.DistanceBetweenTwoPoints(Position, flockSheepIsPartOf.dog.Position);

        float dogDistanceSensitiveMultiplier = (float)DogDistanceSensitiveMultiplier(Config.SheepHowFarAwayItSpotsTheDog, distToPredator);

        // Craig W. Reynold's flocking behavior is controlled by three simple rules: [we add #4, because
        // sheep are scared of dogs, being "wolf" descendants]

        // With these three simple rules, the flock moves in an extremely realistic way, creating complex
        // motion and interaction that would be extremely hard to create otherwise.

        //      _  _   _    ____      _               _                __    _   _                  _   _             
        //    _| || |_/ |  / ___|___ | |__   ___  ___(_) ___  _ __    / /_ _| |_| |_ _ __ __ _  ___| |_(_) ___  _ __  
        //   |_  ..  _| | | |   / _ \| '_ \ / _ \/ __| |/ _ \| '_ \  / / _` | __| __| '__/ _` |/ __| __| |/ _ \| '_ \ 
        //   |_      _| | | |__| (_) | | | |  __/\__ \ | (_) | | | |/ / (_| | |_| |_| | | (_| | (__| |_| | (_) | | | |
        //     |_||_| |_|  \____\___/|_| |_|\___||___/_|\___/|_| |_/_/ \__,_|\__|\__|_|  \__,_|\___|\__|_|\___/|_| |_|
        //                                                                                                            
        // Cohesion: Steer towards average position of neighbours (long range attraction)

        PointF cohesionVector = flockSheepIsPartOf.MoveTowardsCentreOfMass(this);

        //      _  _  ____       _             _     _                             _ _             
        //    _| || ||___ \     / \__   _____ (_) __| |   ___ _ __ _____      ____| (_)_ __   __ _ 
        //   |_  ..  _|__) |   / _ \ \ / / _ \| |/ _` |  / __| '__/ _ \ \ /\ / / _` | | '_ \ / _` |
        //   |_      _/ __/   / ___ \ V / (_) | | (_| | | (__| | | (_) \ V  V / (_| | | | | | (_| |
        //     |_||_||_____| /_/   \_\_/ \___/|_|\__,_|  \___|_|  \___/ \_/\_/ \__,_|_|_| |_|\__, |
        //                                                                                   |___/ 
        // Separation: avoid crowding neighbours (short range repulsion)

        PointF separationVector = flockSheepIsPartOf.MaintainSeparation(this);

        //      _  _  _____      _    _ _               _                     _       _     _                          
        //    _| || ||___ /     / \  | (_) __ _ _ __   | |_ ___    _ __   ___(_) __ _| |__ | |__   ___  _   _ _ __ ___ 
        //   |_  ..  _||_ \    / _ \ | | |/ _` | '_ \  | __/ _ \  | '_ \ / _ \ |/ _` | '_ \| '_ \ / _ \| | | | '__/ __|
        //   |_      _|__) |  / ___ \| | | (_| | | | | | || (_) | | | | |  __/ | (_| | | | | |_) | (_) | |_| | |  \__ \
        //     |_||_||____/  /_/   \_\_|_|\__, |_| |_|  \__\___/  |_| |_|\___|_|\__, |_| |_|_.__/ \___/ \__,_|_|  |___/
        //                                |___/                                 |___/                                  
        // Alignment: steer towards average heading of neighbours

        PointF alignmentVector = flockSheepIsPartOf.MatchVelocityOfNearbySheep(this);

        //      _  _   _  _        _             _     _   ____               _       _                 
        //    _| || |_| || |      / \__   _____ (_) __| | |  _ \ _ __ ___  __| | __ _| |_ ___  _ __ ___ 
        //   |_  ..  _| || |_    / _ \ \ / / _ \| |/ _` | | |_) | '__/ _ \/ _` |/ _` | __/ _ \| '__/ __|
        //   |_      _|__   _|  / ___ \ V / (_) | | (_| | |  __/| | |  __/ (_| | (_| | || (_) | |  \__ \
        //     |_||_|    |_|   /_/   \_\_/ \___/|_|\__,_| |_|   |_|  \___|\__,_|\__,_|\__\___/|_|  |___/
        //                                                                                                      
        // Escape: make them move away from the dog

        PointF escapeVector = Flock.EscapeFromTheDog(this, flockSheepIsPartOf.dog.Position);

        // Steer towards a chosen point.
        PointF guidanceVector = new(0, 0); // Flock.EncourageDirectionTowards(new PointF(10, 10), this); | Flock.EncourageDirectionAway(new PointF(10, 10), this);

        // Add a wind or flow of river
        PointF windVector = Flock.Wind();

        // apply a "velocity" to our sheep horizontally and vertically based on the "rules"
        Velocity.X += Config.SheepMultiplierCohesion * (1 + dogDistanceSensitiveMultiplier * Config.SheepMultiplierCohesionThreatenedByDog) * cohesionVector.X +
                      Config.SheepMultiplierSeparation * (1 + dogDistanceSensitiveMultiplier * Config.SheepMultiplierSeparationThreatenedByDog) * separationVector.X +
                      Config.SheepMultiplierAlignment * (1 + dogDistanceSensitiveMultiplier * Config.SheepMultiplierAlignmentThreatenedByDog) * alignmentVector.X +
                      windVector.X +
                      Config.SheepMultiplierGuidance * (1 + dogDistanceSensitiveMultiplier * Config.SheepMultiplierGuidanceThreatenedByDog) * guidanceVector.X +
                      Config.SheepMultiplierEscape * escapeVector.X;

        Velocity.Y += Config.SheepMultiplierCohesion * (1 + dogDistanceSensitiveMultiplier * Config.SheepMultiplierCohesionThreatenedByDog) * cohesionVector.Y +
                      Config.SheepMultiplierSeparation * (1 + dogDistanceSensitiveMultiplier * Config.SheepMultiplierSeparationThreatenedByDog) * separationVector.Y +
                      Config.SheepMultiplierAlignment * (1 + dogDistanceSensitiveMultiplier * Config.SheepMultiplierAlignmentThreatenedByDog) * alignmentVector.Y +
                      windVector.Y +
                      Config.SheepMultiplierGuidance * (1 + dogDistanceSensitiveMultiplier * Config.SheepMultiplierGuidanceThreatenedByDog) * guidanceVector.Y +
                      Config.SheepMultiplierEscape * escapeVector.Y;


        // This final velocity vector is capped to a certain value vmax that represents the sheep’s maximum velocity. 
        // vmax increases as the predator approaches. If the final velocity vector is below a certain threshold vmin 
        // it is set to zero. The vector is also set to zero if it is directed at a point behind the sheep, as the 
        // sheep can only turn at a certain angular velocity.        

        // velocity drives both desired speed and angle.
        // but a sheep moves forward, not sideways; so we convert desired velocity into a reasonable angle
        double angle = Math.Atan2(Velocity.Y, Velocity.X);

        Angle = (float)angle.Clamp(Angle - 0.0872665 / 2, Angle + 0.0872665 / 2);

        StopSheepRunningUnrealisticallyFast();

        Position.X += Velocity.X;
        Position.Y += Velocity.Y;

        // ensure the sheep doesn't move off screen.
        PointF adjustmentVectorToKeepSheepWithinSheepPen = flockSheepIsPartOf.EnforceBoundary(this);

        Position.X += adjustmentVectorToKeepSheepWithinSheepPen.X;
        Position.Y += adjustmentVectorToKeepSheepWithinSheepPen.Y;

        // occasionally let them rest, eat and poop
        //if (RandomNumberGenerator.GetInt32(0, 500) > 492) MakeSheepPause();
    }

    /// <summary>
    /// Stops the sheep going too fast.
    /// </summary>
    internal void StopSheepRunningUnrealisticallyFast()
    {
        // Pythagoras to turn x & y velocity into a diagonal velocity
        float velocity = (float)Math.Sqrt(Math.Pow(Velocity.X, 2) + Math.Pow(Velocity.Y, 2));

        // no adjustment required?
        if (velocity < Config.SheepMaximumVelocityInAnyDirection)
        {
            // stop the sheep if it has "tiny" velocity
            if (Math.Abs(velocity) < Config.SheepMinimumSpeedBeforeStop) Velocity.X = 0;
            if (Math.Abs(velocity) < Config.SheepMinimumSpeedBeforeStop) Velocity.Y = 0;

            // no adjustment 
            return;
        }

        // we need to reduce the speed
        Velocity.X = Velocity.X / velocity * Config.SheepMaximumVelocityInAnyDirection;

        Velocity.Y = Velocity.Y / velocity * Config.SheepMaximumVelocityInAnyDirection;
    }

    /// <summary>
    /// Draw the sheep in the sheep pen.
    /// 
    /// Initially we create as a filled in white circle, maybe later get a little more fancy.
    /// </summary>
    /// <param name="g"></param>
    internal void Draw(Graphics g, Color colour)
    {
        // draw the sheep blob, a white blob with black dot for head
        g.FillEllipse(new SolidBrush(colour), Position.X - 3, Position.Y - 3, 5, 5);

        float x = (float)Math.Cos(Angle) * 3 + Position.X;
        float y = (float)Math.Sin(Angle) * 3 + Position.Y;

        // black head
        g.DrawRectangle(Pens.Black, x, y, 1, 1);

#if drawVelocityArrows
        // velocity arrows
        double angle = Angle;
        float size = (float)Math.Sqrt(Math.Pow(Velocity.X, 2) + Math.Pow(Velocity.Y, 2));

        using Pen p2 = new(Color.DarkSalmon);
        p2.DashStyle = DashStyle.Dot;
        p2.EndCap = LineCap.ArrowAnchor;
        g.DrawLine(p2,
                   (int)Position.X, (int)Position.Y,
                   (int)(20 * size * Math.Cos(angle) + Position.X),
                   (int)(20 * size * Math.Sin(angle) + Position.Y));
#endif

#if drawCircleAroundCenterOfMass
        using Pen pen = new(Color.FromArgb(100, 255, 255, 255));
        pen.DashStyle = System.Drawing.Drawing2D.DashStyle.Dash;

        g.DrawEllipse(pen, Position.X - Config.SheepCloseEnoughToBeAMass, Position.Y - Config.SheepCloseEnoughToBeAMass, Config.SheepCloseEnoughToBeAMass * 2, Config.SheepCloseEnoughToBeAMass * 2);
#endif
    }
}