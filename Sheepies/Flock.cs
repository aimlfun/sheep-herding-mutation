//#define PunishingStragglers
//#define DrawCircleAroundCoM
using SheepHerderMutation;
using SheepHerderMutation.Configuration;
using SheepHerderMutation.Predator;
using SheepHerderMutation.Utilities;
using System.Diagnostics;
using System.Drawing.Drawing2D;
using System.Security.Cryptography;
using System.Windows.Forms;

namespace SheepHerderMutation.Sheepies;

/// <summary>
/// Class represents a flock of "sheep". 
/// 
/// Flocking: https://en.wikipedia.org/wiki/Flocking_(behavior)
/// 
/// "Flocking" is the collective motion by a group of self-propelled entities and is a collective animal behavior 
/// exhibited by many living beings such as birds, fish, bacteria, and insects.
/// It is considered an emergent behavior arising from simple rules that are followed by individuals and does not 
/// involve any central coordination.
///  
/// https://vergenet.net/~conrad/boids/pseudocode.html article helped a lot to become proficient at flocking.
/// </summary>
internal class Flock
{
    //   _____ _            _    
    //  |  ___| | ___   ___| | __
    //  | |_  | |/ _ \ / __| |/ /
    //  |  _| | | (_) | (__|   < 
    //  |_|   |_|\___/ \___|_|\_\

    #region CONSTANTS
    /// <summary>
    /// When this is exceeded, the UI jumps out of quiet mode upon reaching. At that point you've
    /// missed what it did, but you know it attained it.
    /// </summary>
    internal const int c_exitQuietModeWhenScoreReached = 500; // 5 sheep
    #endregion

    /// <summary>
    /// How many sheep are in the scoring zone (the box top right
    /// </summary>
    internal int numberOfSheepInPenZone = 0;

    /// <summary>
    /// Indirect link to dog and its brain. This is the "Id" of the neural network.
    /// </summary>
    internal readonly int Id;

    /// <summary>
    /// Coloured sheep? A bit too much dye.
    /// </summary>
    internal Color sheepColour = Color.Pink;

    /// <summary>
    /// Horizontal confines of the sheep pen in pixel.
    /// </summary>
    internal int SheepPenWidth;

    /// <summary>
    /// Vertical confines of the sheep pen in pixel.
    /// </summary>
    internal int SheepPenHeight;

    /// <summary>
    /// Represents the flock of sheep.
    /// </summary>
    internal readonly List<Sheep> flock = new();

    /// <summary>
    /// The dog chasing the flock.
    /// </summary>
    internal readonly Dog dog;

    /// <summary>
    /// Next location we want the flock to go towards.
    /// </summary>
    private int nextWayPointToHeadTo = 0;

    /// <summary>
    /// Next location we want the flock to go towards.
    /// But tracks "time" it was set, so we can ensure generations progress or halt.
    /// </summary>
    internal int NextWayPointToHeadTo
    {
        get
        {
            return nextWayPointToHeadTo;
        }

        set
        {
            if (nextWayPointToHeadTo == value) return;

            nextWayPointToHeadTo = value;

            timeOfLastCheckpoint = DateTime.Now; // so we can kill time wasters
        }
    }

    /// <summary>
    /// Failed to go towards way point - make the flock in "failed" state (doesn't move).
    /// </summary>
    internal bool flockIsFailure = false;

    /// <summary>
    /// Used to show WHY this flock was terminated early.
    /// </summary>
    internal string failureReason = "RUNNING";

    /// <summary>
    /// Used to know if the sheep are not progressing thru checkpoints in a reasonable time. Without
    /// it will increase the time indefinitely and not exit to next generation.
    /// </summary>
    DateTime timeOfLastCheckpoint = DateTime.Now;

    /// <summary>
    /// This is the green field with brown hedges/fences, and a score zone. 
    /// We paint it once, and use it as the basis of our background.
    /// </summary>
    internal Bitmap s_backgroundImage;

    internal Bitmap? cachedImageAfterDeath = null;

    /// <summary>
    /// Constructor: Creates a flock of sheep.
    /// </summary>
    internal Flock(int id, List<PointF> sheepPositions, int width, int height)
    {
        Id = id;
        SheepPenWidth = width;
        SheepPenHeight = height;

        // random
        sheepColour = Color.White; // Color.FromArgb(RandomNumberGenerator.GetInt32(0, 255), RandomNumberGenerator.GetInt32(0, 255), RandomNumberGenerator.GetInt32(0, 255));

        // add the required number of sheep
        for (int i = 0; i < Config.InitialFlockSize; i++)
        {
            flock.Add(new Sheep(this, sheepPositions[i]));
        }

        // every flock needs a dog.
        dog = new Dog(id, this);

        // if you get this, the cure is simple. To avoid drawing static lines each frame we use a pre-cached copy
        if (LearnToHerd.s_backgroundImage is null) throw new Exception("s_backgroundImage is unpopulated? It should be painted with fences / grass.");

        // for parallel draw, we need multiple images
        s_backgroundImage = new Bitmap(LearnToHerd.s_backgroundImage);
    }

    /// <summary>
    /// Computes center of ALL sheep except this one.
    /// </summary>
    internal PointF CenterOfMassExcludingThisSheep(Sheep thisSheep)
    {
        // center of nothing = nothing
        if (flock.Count < 2) return new PointF(thisSheep.Position.X, thisSheep.Position.Y);

        // compute center
        float x = 0;
        float y = 0;

        foreach (Sheep sheep in flock)
        {
            if (sheep == thisSheep) continue; // center of mass excludes this sheep

            x += sheep.Position.X;
            y += sheep.Position.Y;
        }

        // we exclude "thisSheep", so average is N-1.
        int sheepSummed = flock.Count - 1;

        return new PointF(x / sheepSummed, y / sheepSummed);
    }

    /// <summary>
    /// Returns angle sheep need to go to, to be at next waypoint.
    /// Called everytime we move the predator to know where we
    /// require the sheep to go (feeds as input).
    /// </summary>
    /// <returns></returns>
    internal double AngleToNextWayPoint()
    {
        PointF centreOfMass = TrueCentreOfMass(out bool stragglers);
        PointF wayPointToHeadTowards;

#if PunishingStragglers
        // because our AI is basic, stragglers mean we will not complete the required task (sheep in pen)
        if (stragglers)
        {
            failureReason = "STRAGGLERS";
            flockIsFailure = true;
        }
#endif
        // flock is out of sight to predator
        if (MathUtils.DistanceBetweenTwoPoints(dog.Position, centreOfMass) > Config.DogSensorOfSheepVisionDepthOfVisionInPixels * 1.4F) // +0.4, because distance is to CoM
        {
            failureReason = "OUT OF SIGHT";
            flockIsFailure = true;
        }
        else
        {
            AdjustClosestWayPointForwards(centreOfMass);
        }

        wayPointToHeadTowards = LearnToHerd.s_wayPointsSheepNeedsToGoThru[NextWayPointToHeadTo];

        double angle = Math.Atan2(wayPointToHeadTowards.Y - centreOfMass.Y,
                                  wayPointToHeadTowards.X - centreOfMass.X);

        TimeSpan timePassed = DateTime.Now.Subtract(timeOfLastCheckpoint);

        // without this the dog could be stationary and sheep not move; this ensures we end the generation if the
        // dog is uncooperative.
        if (timePassed.TotalSeconds > Config.DogMaximumTimeWastingAllowedInSeconds)
        {
            if (NextWayPointToHeadTo < LearnToHerd.s_wayPointsSheepNeedsToGoThru.Length - 1) // 24 is exception, it's where the pen is
                failureReason = "TIME WASTING";
            else
                failureReason = "TIME UP";

            flockIsFailure = true;
        }

        return angle;
    }

    /// <summary>
    /// Sets nextWayPointToHeadTo.
    /// </summary>
    /// <param name="centreOfMass"></param>
    private void AdjustClosestWayPointForwards(PointF centreOfMass)
    {
        NextWayPointToHeadTo = GetClosestWayPointForwards(centreOfMass, NextWayPointToHeadTo);
    }

    /// <summary>
    /// Sets nextWayPointToHeadTo.
    /// </summary>
    /// <param name="centreOfMass"></param>
    public static int GetClosestWayPointForwards(PointF centreOfMass, int currentWayPoint)
    {
        // Determine the *furthest* waypoint based on where the centre of mass is (without going thru a wall). 
        // This means it takes a short-cut where possible, rather than following dots.

        int closestWayPointByIndex = -1;
        float closestDistanceToWayPoint = -1;
        // evaluate all check points close to the current one.
        int minWayPointIndex = Math.Max(currentWayPoint, 0);
        int maxWayPointIndex = LearnToHerd.s_wayPointsSheepNeedsToGoThru.Length;

        for (int indexOfWayPoints = minWayPointIndex; indexOfWayPoints < maxWayPointIndex; indexOfWayPoints++)
        {
            // check one by one, trying to find the furthest in range
            Point wayPointForIndex = LearnToHerd.s_wayPointsSheepNeedsToGoThru[indexOfWayPoints];

            float distanceFromCenterOfMassToWayPoint = MathUtils.DistanceBetweenTwoPoints(wayPointForIndex, centreOfMass);

            if (!RouteHasFencesInBetweenFlockAndWayPoint(centreOfMass, wayPointForIndex) && distanceFromCenterOfMassToWayPoint > closestDistanceToWayPoint)
            {
                closestWayPointByIndex = indexOfWayPoints;
                closestDistanceToWayPoint = distanceFromCenterOfMassToWayPoint;
            }
        }

        if (closestWayPointByIndex > 0) currentWayPoint = closestWayPointByIndex;

        if (currentWayPoint >= LearnToHerd.s_wayPointsSheepNeedsToGoThru.Length) --currentWayPoint;

        return currentWayPoint;
    }

    /// <summary>
    /// Check the fences, to see if the line that the fence is declared using intersects with a line between flock COM and waypoint.
    /// </summary>
    /// <param name="flockCentreOfMass"></param>
    /// <param name="wayPoint"></param>
    /// <returns></returns>
    private static bool RouteHasFencesInBetweenFlockAndWayPoint(PointF flockCentreOfMass, PointF wayPoint)
    {
        /*                         x wayPoint
         *                        .
         *                       .
         *                    =============== fence
         *                     .
         *                    .
         *                   .
         *                  o flock CoM
         */

        foreach (PointF[] points in LearnToHerd.s_lines)
        {
            // s_lines is an array of *joined* points (that we draw lines between). We thus have
            // to take one line at a time and check..
            for (int i = 0; i < points.Length - 1; i++) // -1, because we're doing line "i" to "i+1"
            {
                PointF point1 = points[i];
                PointF point2 = points[i + 1];

                // will the sheep need to go thru a wall to get to the wayPoint?
                if (MathUtils.GetLineIntersection(point1, point2, flockCentreOfMass, wayPoint, out _ /* we don't care *where* it intersects */))
                {
                    return true; // lines intersect
                }
            }
        }

        return false;
    }
    /// <summary>
    /// Center of mass of ALL the sheep (not excluding one)
    /// </summary>
    /// <returns></returns>
    internal PointF TrueCentreOfMass(out bool stragglers)
    {
        // compute center using average of x & y
        float x = 0;
        float y = 0;

        foreach (Sheep sheep in flock)
        {
            x += sheep.Position.X;
            y += sheep.Position.Y;
        }

        PointF com = new(x / flock.Count, y / flock.Count);

        // detect stragglers (those too far from CoM)
        stragglers = false;

        foreach (Sheep sheep in flock)
        {
            if (MathUtils.DistanceBetweenTwoPoints(com, sheep.Position) > 150) stragglers = true;
        }

        // centre of ALL sheep
        return com;
    }

    /// <summary>
    /// Localised, single wind for all sheep. If it were different per sheep they would fast
    /// become a misaligned mess.
    /// </summary>
    /// <param name="b"></param>
    /// <returns></returns>
    internal static PointF Wind()
    {
        return new(0, 0);
    }

    /// <summary>
    /// Tendency towards a particular place
    /// For example, to steer a sparse flock of sheep or cattle to a narrow gate. Upon reaching this point, the goal for a particular
    /// sheep could be changed to encourage it to move away to make room for other members of the flock. 
    /// Note that if this 'gate' is flanked by impenetrable objects as accounted for in Rule 2 above, then the flock will realistically mill around the gate and slowly trickle through it.
    /// </summary>
    /// <param name="desiredPlace"></param>
    /// <param name="thisSheep"></param>
    /// <returns></returns>
    internal static PointF EncourageDirectionTowards(PointF desiredPlace, Sheep thisSheep)
    {
        // Note that this rule moves the boid 1 % of the way towards the goal at each step. Especially for distant goals,
        // we limit the magnitude of the returned vector.
        return new(((desiredPlace.X - thisSheep.Position.X) / 100).Clamp(-Config.SheepMaximumVelocityInAnyDirection / 2, Config.SheepMaximumVelocityInAnyDirection / 2),
                   ((desiredPlace.Y - thisSheep.Position.Y) / 100).Clamp(-Config.SheepMaximumVelocityInAnyDirection / 2, Config.SheepMaximumVelocityInAnyDirection / 2));
    }

    /// <summary>
    /// Prevent sheep moving outside the sheep pen / off-screen.
    /// </summary>
    /// <param name="thisSheep"></param>
    /// <returns></returns>
    internal PointF EnforceBoundary(Sheep thisSheep)
    {
        PointF p = new(0, 0);

        if (thisSheep.Position.X < 5) p.X = 4;

        if (thisSheep.Position.X > SheepPenWidth - 5) p.X = -4;

        if (thisSheep.Position.Y < 5) p.Y = 4;

        if (thisSheep.Position.Y > SheepPenHeight - 5) p.Y = -4;

        return p;
    }

    #region SHEEP RULES OF MOVEMENT
    /// <summary>
    /// Rule 1: Sheep like to move towards the centre of mass of flock.
    /// </summary>
    /// <param name="thisSheep"></param>
    /// <returns></returns>
    internal PointF MoveTowardsCentreOfMass(Sheep thisSheep)
    {
        PointF pointF = CenterOfMassExcludingThisSheep(thisSheep);

        // move it 1% of the way towards the centre
        return new PointF((pointF.X - thisSheep.Position.X) / 100,
                          (pointF.Y - thisSheep.Position.Y) / 100);
    }

    /// <summary>
    /// Rule 2: Boids try to keep a small distance away from other objects (including other boids).
    /// </summary>
    /// <param name="thisSheep"></param>
    /// <returns></returns>
    internal PointF MaintainSeparation(Sheep thisSheep)
    {
        // The purpose of this rule is to for boids to make sure they don't collide into each other.
        // I simply look at each boid, and if it's within a defined small distance (say 100 units) of
        // another boid move it as far away again as it already is. This is done by subtracting from a
        // vector c the displacement of each boid which is near by.
        // We initialise c to zero as we want this rule to give us a vector which when added to the
        // current position moves a boid away from those near it.

        const float c_size = 6;

        // We initialise c to zero as we want this rule to give us a vector which when added to the
        // current position moves a boid away from those near it.
        PointF separationVector = new(0, 0);

        // collision with another sheep?
        foreach (Sheep sheep in flock)
        {
            if (sheep == thisSheep) continue; // excludes this sheep

            if (MathUtils.DistanceBetweenTwoPoints(sheep.Position, thisSheep.Position) < c_size)
            {
                separationVector.X -= sheep.Position.X - thisSheep.Position.X;
                separationVector.Y -= sheep.Position.Y - thisSheep.Position.Y;
            }
        }

        // collision with any walls?
        foreach (PointF[] points in LearnToHerd.s_lines)
        {
            for (int i = 0; i < points.Length - 1; i++) // -1, because we're doing line "i" to "i+1"
            {
                PointF point1 = points[i];
                PointF point2 = points[i + 1];

                // touched wall? returns the closest point on the line to the sheep. We check the distance
                if (MathUtils.IsOnLine(point1, point2, new PointF(thisSheep.Position.X + separationVector.X, thisSheep.Position.Y + separationVector.Y), out PointF closest)
                    && MathUtils.DistanceBetweenTwoPoints(closest, thisSheep.Position) < 9)
                {
                    // yes, need to back off from the wall
                    separationVector.X -= 2.5f * (closest.X - thisSheep.Position.X);
                    separationVector.Y -= 2.5f * (closest.Y - thisSheep.Position.Y);
                }
            }
        }

        return separationVector; // how much to separate this sheep
    }

    /// <summary>
    /// Tries to provide a vector to escape the dog.
    /// </summary>
    /// <param name="sheep"></param>
    /// <param name="dog"></param>
    /// <returns></returns>
    internal static PointF EscapeFromTheDog(Sheep sheep, PointF dog)
    {
        float x = sheep.Position.X - dog.X;
        float y = sheep.Position.Y - dog.Y;

        float distToPredator = (float)Math.Sqrt(Math.Pow(x, 2) + Math.Pow(y, 2)) + .00001f;

        return new(x / distToPredator * InverseSquare(distToPredator, 10),
                   y / distToPredator * InverseSquare(distToPredator, 10));
    }

    /// <summary>
    /// Inverse Square Function
    /// In two of the rules, Separation and Escape, nearby objects are prioritized higher than
    /// those further away.This prioritization is described by an inverse square function.
    /// This function, seen in Formula (3.3), is referred to as inv throughout the text.
    /// </summary>
    /// <param name="x">is the distance between the objects</param>
    /// <param name="s">s is a softness factor that slows down the rapid decrease of the function value | s = 1 for Separation and s = 10 for Escape.</param>
    /// <returns></returns>
    private static float InverseSquare(float x, float s)
    {
        float e = 0.0000000001f; // is a small value used to avoid division by zero, when x = 0.
        return (float)Math.Pow(x / (s + e), -2);
    }

    /// <summary>
    /// Rule 3: Boids try to match velocity with near boids.
    /// </summary>
    /// <param name="thisSheep"></param>
    /// <returns></returns>
    internal PointF MatchVelocityOfNearbySheep(Sheep thisSheep)
    {
        // This is similar to Rule 1, however instead of averaging the positions of the other boids
        // we average the velocities. We calculate a 'perceived velocity', pvJ, then add a small portion
        // (about an eighth) to the boid's current velocity.

        PointF c = new(0, 0);
        int countSheep = 0;

        foreach (Sheep sheep in flock)
        {
            if (sheep == thisSheep) continue; // excludes this sheep

            /* The alignment rule is calculated for each sheep s. Each sheep si within a radius of
                50 pixels has a velocity siv that contributes equally to the final rule vector. The size
                of the rule vector is determined by the velocity of all nearby sheep N. The vector is
                directed in the average direction of the nearby sheep. The rule vector is calculated
                with the function .
            */

            if (MathUtils.DistanceBetweenTwoPoints(sheep.Position, thisSheep.Position) > Config.SheepCloseEnoughToBeAMass) continue;

            ++countSheep;

            c.X += sheep.Velocity.X;
            c.Y += sheep.Velocity.Y;
        }

        if (countSheep > 0)
        {

            c.X /= countSheep;
            c.Y /= countSheep;
        }

        return new PointF((c.X - thisSheep.Velocity.X) / 8, (c.Y - thisSheep.Velocity.Y) / 8);
    }
    #endregion

    /// <summary>
    /// Moves and draws the flock of sheep.
    /// </summary>
    internal void Move()
    {
        if (flockIsFailure) return;

        numberOfSheepInPenZone = 0;

        // move them all, using Reynolds swarm mathematics
        foreach (Sheep s in flock)
        {
            s.Move();

            if (LearnToHerd.s_sheepPenScoringZone.Contains(s.Position)) ++numberOfSheepInPenZone; // track the number in the pen
        }

        dog.Move();
    }

    /// <summary>
    /// Draws the sheep and predator.
    /// </summary>
    /// <param name="g"></param>
    internal void Draw(Graphics g)
    {
        ColourInTheWayPointsTheSheepHaveGoneCloseTo(g);

        // compute center of all sheep whilst drawing the sheep

        float x = 0;
        float y = 0;

        // draw each sheep
        foreach (Sheep sheep in flock)
        {
            sheep.Draw(g, flockIsFailure ? Color.FromArgb(50, 255, 255, 255) : sheepColour);

            x += sheep.Position.X;
            y += sheep.Position.Y;
        }

        // calculate centre of ALL sheep (known as center of mass)
        PointF centerOfMass = new(x / flock.Count, y / flock.Count);

        DrawXatCenterOfMass(g, centerOfMass);
        DrawCircleAroundCenterOfMass(g, centerOfMass);
        DrawPointerToNextWayPointFromCenterOfMass(g, centerOfMass);

        dog.Draw(g);
    }

    /// <summary>
    /// Returns the angle the flock is moving in radians
    /// </summary>
    /// <returns></returns>
    internal double GetAngleFlockIsMovingInRadians()
    {
        float x = 0, y = 0;

        foreach (Sheep sheep in flock)
        {
            x += sheep.Velocity.X;
            y += sheep.Velocity.Y;
        }

        // ArcTan gives us the angle.
        return Math.Atan2(y, x);
    }

    /// <summary>
    /// Debugging requires us to know where the center of mass is. So we draw a red "X".
    /// </summary>
    /// <param name="g"></param>
    /// <param name="centerOfMass"></param>
    private static void DrawXatCenterOfMass(Graphics g, PointF centerOfMass)
    {
        // x marks the spot for center of mass
        g.DrawLine(Pens.Red, centerOfMass.X - 4, centerOfMass.Y - 4, centerOfMass.X + 4, centerOfMass.Y + 4);
        g.DrawLine(Pens.Red, centerOfMass.X - 4, centerOfMass.Y + 4, centerOfMass.X + 4, centerOfMass.Y - 4);
    }

    /// <summary>
    /// Way points are faint blobs. As the sheep go near the blobs we register it by
    /// drawing them in a lighter colour.
    /// </summary>
    /// <param name="g"></param>
    private void ColourInTheWayPointsTheSheepHaveGoneCloseTo(Graphics g)
    {
        SolidBrush wayPointBrush = new(Color.FromArgb(255, 120, 255, 120));

        for (int i = 0; i < NextWayPointToHeadTo; i++)
        {
            Point wp = LearnToHerd.s_wayPointsSheepNeedsToGoThru[i];

            g.FillEllipse(wayPointBrush, wp.X - 3, wp.Y - 3, 6, 6);
        }

    }

    /// <summary>
    /// Having a circle helps us know where the center of mass ends.
    /// </summary>
    /// <param name="g"></param>
    /// <param name="centerOfMass"></param>
    private static void DrawCircleAroundCenterOfMass(Graphics g, PointF centerOfMass)
    {
#if DrawCircleAroundCoM
        // draw circle around the CoM
        using Pen p = new(Color.FromArgb(50, 255, 50, 50));

        p.DashStyle = DashStyle.Dash;
        g.DrawEllipse(p, new RectangleF(centerOfMass.X - (float)Config.SheepClosenessToMoveToNextWayPoint,
                                        centerOfMass.Y - (float)Config.SheepClosenessToMoveToNextWayPoint,
                                        (float)Config.SheepClosenessToMoveToNextWayPoint * 2,
                                        (float)Config.SheepClosenessToMoveToNextWayPoint * 2));
#endif
    }

    /// <summary>
    /// Pointer to next way point (line with arrow head) from the center of mass.
    /// </summary>
    /// <param name="g"></param>
    /// <param name="centerOfMass"></param>
    /// <param name="wayPointToHeadTowards"></param>
    private void DrawPointerToNextWayPointFromCenterOfMass(Graphics g, PointF centerOfMass)
    {
        PointF wayPointToHeadTowards = LearnToHerd.s_wayPointsSheepNeedsToGoThru[NextWayPointToHeadTo];

        // we have 2 points. ArcTan gives us the angle.
        double angle = Math.Atan2(wayPointToHeadTowards.Y - centerOfMass.Y,
                                   wayPointToHeadTowards.X - centerOfMass.X);

        Pen p2 = new(Color.Black)
        {
            DashStyle = DashStyle.Dot,
            EndCap = LineCap.ArrowAnchor
        };

        // rotate at the angle, origin of the com.
        g.DrawLine(p2, (int)centerOfMass.X, (int)centerOfMass.Y, (int)(30 * Math.Cos(angle) + centerOfMass.X), (int)(30 * Math.Sin(angle) + centerOfMass.Y));
    }

    /// <summary>
    /// Determine how well the sheep have been herded. 
    /// Larger number = better. 
    /// Max score = # sheep * Width of Pen * Height of Pen. 
    /// 
    /// There are lots of ways you could score. We've gone with a "desired" path made of way points and score based on how many
    /// way points they passed. Reaching the pen rewards them handsomely (9000 points per sheep).
    /// </summary>
    /// <returns></returns>
    internal float FitnessScore()
    {
        float score = 0;

        foreach (Sheep s in flock)
        {
            score += LearnToHerd.s_sheepPenScoringZone.Contains(s.Position)
                        ? 100 // a lot higher than us gained from mere way points
                        : NextWayPointToHeadTo;
        }

        score /= flock.Count; // average score for the flock

        if (score > c_exitQuietModeWhenScoreReached)
        {
            LearnToHerd.s_silentMode = false; // wake up out of silent mode
        }

        return score;
    }
}