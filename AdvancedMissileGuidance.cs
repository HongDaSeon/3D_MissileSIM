using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class AdvancedMissileGuidance : MonoBehaviour
{
    public string Property;
    public GameObject Seeker;
    public float NavigationContant;
    public float Speed;
    public GameObject Origin;
    public GameObject Launcher;
    public float EscapePower;
    public float AccelerationLimit;
    public GameObject SimulationMaster;
    public GameObject self;
    public ParticleSystem Explosion;
    public ParticleSystem Smoke;

    public float explosionForce;
    public float explosionRadius;

    public float BurstTime;

    public Collider detonateCollider;

    public string AllieTag;
    public string EnemyTag;

    public bool engaged;


    private Vector3 Omega;
    private Vector3 Rm;
    private Vector3 Vm;

    private Vector3 Rt;
    private Vector3 Vt;

    private Vector3 Rvec;
    private Vector3 Vvec;

    private Vector3 AccCommand;
    private Vector3 AccCommandBody;

    private Rigidbody RB;
    private Matrix4x4 CNB;

    private float psi;
    private float theta;

    private bool is_hit;
    private Vector3 hitPos;
    private GameObject hitTarget;

    private float timeStart;

    // Start is called before the first frame update

    private void OnEnable()
    {
        if (engaged == true)
        {
            transform.position = Launcher.transform.position;
            transform.rotation = Launcher.transform.rotation;

            AccCommand = new Vector3(0f, 0f, 0f);
            RB = gameObject.GetComponent<Rigidbody>();
            //RB.velocity = Speed * transform.forward;

            RB.velocity = (EscapePower * transform.forward);

            float theta = Mathf.Atan2(-Vm.z, Mathf.Sqrt(Mathf.Pow(Vm.x, 2) + Mathf.Pow(Vm.y, 2))) * Mathf.Rad2Deg;
            float psi = Mathf.Atan2(Vm.y, Vm.x) * Mathf.Rad2Deg;

            is_hit = false;
            timeStart = Time.time;
        }
    }

    /*
    void Start()
    {
        
    }
    */
    // Update is called once per frame
    void FixedUpdate()
    {
        if(engaged == true && gameObject.GetComponent<Detonatee>().is_controllable == true)
        {
            CNB = transform.localToWorldMatrix;


            if (Seeker.gameObject.GetComponent<Seeker>().is_lockOn == true)
            {
                RB.velocity = CNB * new Vector3(0, 0, Speed);
            }

            AccCommandBody = CNB.transpose * AccCommand;

            Vector3 SatAcc = new Vector3(Saturation(AccCommandBody.x, -AccelerationLimit, +AccelerationLimit),
                                            Saturation(AccCommandBody.y, -AccelerationLimit, +AccelerationLimit),
                                            0f);

            float dpsiM = SatAcc.x / Speed / Mathf.Cos(theta);
            float dthetaM = -SatAcc.y / Speed;

            transform.Rotate(new Vector3(0, 1, 0), Mathf.Rad2Deg * dpsiM * Time.deltaTime);
            transform.Rotate(new Vector3(1, 0, 0), Mathf.Rad2Deg * dthetaM * Time.deltaTime);

            Rm = RB.position;
            Vm = RB.velocity;

            Quaternion VelRotation = Quaternion.LookRotation(Vm);


            Rt = Seeker.gameObject.GetComponent<Seeker>().TargetPos + new Vector3(0f, 0f, 0f);
            Vt = Seeker.gameObject.GetComponent<Seeker>().TargetVel;

            Rvec = Rt - Rm;
            Vvec = Vt - Vm;

            AccCommand = guidance(Rvec, Vvec, Vm);

            if(is_hit == true)
            {
                transform.position = hitTarget.transform.position;
                transform.rotation = Quaternion.Euler(new Vector3(97f, 11f, 0f));
            }
            else if( Seeker.gameObject.GetComponent<Seeker>().is_lockOn == true)
            {
                hitPos = transform.position;
                hitTarget = Seeker.gameObject.GetComponent<Seeker>().Target.gameObject;
            }

            if((Time.time - timeStart) > BurstTime)
            {
                selfDetonation();
                engaged = false;
            }
        }

    }


    Vector3 guidance(Vector3 Rvec, Vector3 Vvec, Vector3 Vm)
    {
        Omega = Vector3.Cross(Rvec, Vvec) / Vector3.Dot(Rvec, Rvec);
        return NavigationContant * Vector3.Cross(Omega, Vm);
    }

    float Saturation(float val, float minn, float maxx)
    {
        return val * ((minn < val & val < maxx) ? 1f : 0f) + minn * ((minn >= val) ? 1f : 0f) + maxx * ((val >= maxx) ? 1f : 0f);
    }

    public void Detonator(Collision HitObject)
    {

        HitObject.gameObject.GetComponent<Detonatee>().detonate(true);
        selfDetonation();
        transform.position = hitTarget.transform.position;
        RB.isKinematic = true;
        detonateCollider.isTrigger = true;

        HitObject.gameObject.GetComponent<Rigidbody>().AddExplosionForce(explosionForce, transform.position, explosionRadius);

        is_hit = true;

    }

    public void selfDetonation()
    {
        gameObject.GetComponent<Detonatee>().detonate(false);
    }

    public void Launch()
    {
        engaged = true;
        //Start();
        OnEnable();       
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.tag == EnemyTag)
        {
            Detonator(collision);
        }
        
    }
}
