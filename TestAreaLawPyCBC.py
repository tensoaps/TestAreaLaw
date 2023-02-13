import os, sys
import numpy as np
import matplotlib.pyplot as plt

run_code = 1 
use_server = 0 
plot_result = 0
use_ratio = False
useGatedModel = True
sampler_name = ['dynesty', 'pymultinest'][0]
sample_method = ['rwalk', 'rslice'][0]
if use_server:
    pool_size, npoints = [8, 16], 2000
else:
    pool_size, npoints = [1, 1], 400
task_num1 = int(sys.argv[1])
task_num2 = int(sys.argv[2])

outdir = 'AreaLaw'
if useGatedModel:
    label_list = ['inje1_insp_gate_O4', 'inje1_insp_gate_O5', \
                  'inje2_insp_gate_O4', 'inje2_insp_gate_O5', \
                  'inje3_insp_gate_O4', 'inje3_insp_gate_O5', \
                  'inje4_insp_gate_O4', 'inje4_insp_gate_O5', \
                  'inje5_insp_gate_O4', 'inje5_insp_gate_O5']
else:
    label_list = ['inje1_full_IMR_O4', 'inje1_full_IMR_O5', \
                  'inje2_full_IMR_O4', 'inje2_full_IMR_O5', \
                  'inje3_full_IMR_O4', 'inje3_full_IMR_O5', \
                  'inje4_full_IMR_O4', 'inje4_full_IMR_O5', \
                  'inje5_full_IMR_O4', 'inje5_full_IMR_O5']

event_list = [[True, True], [True, False], [False, True]]
event_idx, pool_size = event_list[0], pool_size[task_num1%2]

seedbank = []
np.random.seed(31415926)
seed_list = np.random.randint(1e5, 1e6, 5)
for seed in seed_list:
    np.random.seed(seed)
    seedbank.append([np.random.randint(1e5, 1e6, 5), np.random.randint(1e5, 1e6, 5)])
seedlist = seedbank[task_num2]
label = label_list[task_num1]+'_seed'+sys.argv[2]

duration = 8
delta_f = 1./duration
sampling_frequency = 4096
f_nqy = sampling_frequency/2
delta_t = 1./sampling_frequency
psd_length = int(f_nqy*duration)+1
f_lower, f_upper, f_ref = 15, 2048, 20
template_list = ['IMRPhenomXPHM', 'IMRPhenomPv3HM']
template_name = template_list[0]
label = label+'_'+template_name
fcut_list, gate_time_list = [], []

if run_code:
    import bilby
    from bilby.core.utils import setup_logger
    setup_logger(log_level=['info', 'warning'][0])
    det_names = ['H1', 'L1', 'V1', 'K1', 'I1']

    load_utils = 1
    if load_utils:
        from bilby.gw.conversion import bilby_to_lalsimulation_spins
        def transform_bilby_to_pycbc(parameters, f_ref=20, return_iota=False):
            iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bilby_to_lalsimulation_spins(
                theta_jn=parameters['theta_jn'], phi_jl=parameters['phi_jl'], tilt_1=parameters['tilt_1'], \
                tilt_2=parameters['tilt_2'], phi_12=parameters['phi_12'], a_1=parameters['a_1'], \
                a_2=parameters['a_2'], mass_1=parameters['mass_1'], mass_2=parameters['mass_2'], \
                reference_frequency=f_ref, phase=parameters['phase'])
            pycbc_vari_name = ['mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y', 'spin2z', \
                               'distance', 'inclination', 'tc', 'coa_phase', 'polarization', 'ra', 'dec']
            pycbc_parameters = dict(zip(pycbc_vari_name, [parameters['mass_1'], parameters['mass_2'], \
                float(spin_1x), float(spin_1y), float(spin_1z), float(spin_2x), float(spin_2y), float(spin_2z), \
                parameters['luminosity_distance'], parameters['theta_jn'], parameters['geocent_time'], \
                parameters['phase'], parameters['psi'], parameters['ra'], parameters['dec']]))
            if return_iota:
                pycbc_parameters.update({'inclination':float(iota)})
            return pycbc_parameters

        class MultiEventLikelihood(bilby.Likelihood):
            def __init__(self, likelihood_list, parameters_list, share_par_names, \
                priors, f_ref, event_tag_list=['1g_', '2g_'], pycbc_params=None):
                parameters = dict.fromkeys(priors.keys())
                bilby.Likelihood.__init__(self, parameters=parameters)
                self.event_tag_list = event_tag_list
                if pycbc_params is not None:
                    self.waveform_params = dict(zip(parameters_list, pycbc_params))
                else:
                    self.waveform_params = dict(zip(parameters_list, parameters_list))
                self.f_ref = f_ref
                self.likelihood_list = likelihood_list
                self.parameters_list = parameters_list
                self.share_par_names = share_par_names
            def log_likelihood(self):
                llh = 0
                for i,log_like_func in enumerate(self.likelihood_list):
                    parameters = {}
                    for key in self.parameters_list:
                        if key in self.share_par_names:
                            parameters.update({self.waveform_params[key]:self.parameters[key]})
                        else:
                            parameters.update({self.waveform_params[key]:self.parameters[self.event_tag_list[i]+key]})
                    log_like_func.parameters.update(parameters)
                    llh += log_like_func.log_likelihood()
                return llh
            def noise_log_likelihood(self):
                llh = 0
                for i,log_like_func in enumerate(self.likelihood_list):
                    llh += log_like_func.noise_log_likelihood()
                return llh


        from pycbc.detector import Detector
        from pycbc.pnutils import hybrid_meco_frequency
        from pycbc.waveform.utils import time_from_frequencyseries, td_taper
        from pycbc.inference.models import GaussianNoise, CallModel, GatedGaussianNoise
        from pycbc.waveform.generator import FDomainDetFrameGenerator, FDomainCBCGenerator
        from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q # pycbc use q=m1/m2
        def calculate_thMECO(detframe_waveforms, f_hMECO, parameters, start_time, f_lower, duration, det='H1'):
            flower_idx = int((f_lower+1)*duration)
            wfs_det = detframe_waveforms[det]
            freq_array = wfs_det.sample_frequencies[flower_idx:].numpy()
            fhMECO_idx = np.where(freq_array <= f_hMECO)[0][-1]
            t_from_freq = time_from_frequencyseries(\
                wfs_det[flower_idx:], sample_frequencies=freq_array)
            t_hMECO = t_from_freq[fhMECO_idx]
            tdelay = Detector(det).time_delay_from_earth_center(\
                parameters['ra'], parameters['dec'], parameters['tc'])
            return t_hMECO-tdelay+start_time+duration

        class InspiralLikelihood(bilby.Likelihood):
            def __init__(self, det_names, start_time, duration, strain_data, psds, gate_time_start, useGatedModel=False, **kwargs):
                super(InspiralLikelihood, self).__init__(dict())
                self.psds = psds
                self.det_names = det_names
                waveform_params = ['mass1', 'mass2', 'spin1x', 'spin1y', 'spin1z', 'spin2x', 'spin2y', 'spin2z', \
                                   'distance', 'inclination', 'tc', 'coa_phase', 'polarization', 'ra', 'dec']
                self.static_params = kwargs
                self.FDGen_model = FDomainDetFrameGenerator(FDomainCBCGenerator, epoch=start_time, \
                    variable_args=waveform_params, detectors=det_names, delta_f=1./duration, **self.static_params)
                self.useGatedModel = useGatedModel
                if strain_data is not None:
                    f_lower = {det: self.static_params['f_lower'] for det in det_names}
                    if self.useGatedModel:
                        self.gate_parames = {'t_gate_start':gate_time_start, 't_gate_end': start_time+duration}
                        self.model = GatedGaussianNoise(waveform_params, strain_data, f_lower, psds=psds, static_params=self.static_params)
                        self.model.update(**self.gate_parames)
                        self.lognl = self.model._lognl()
                        # insert the following codes to gate_and_paint()
                        #invpsd = invpsd.copy()
                        #bad_values = invpsd.numpy()
                        #bad_length = len(numpy.where(numpy.isnan(bad_values)
                        #    +(~numpy.isfinite(bad_values)))[0])
                        #invpsd[0:bad_length] = 0
                    else:
                        self.model = GaussianNoise(waveform_params, strain_data, f_lower, psds=psds, static_params=self.static_params)
                        self.lognl = self.model.lognl
                    self.optimalSNR = {}
                    self.get_optimalSNR = False
            def calculate_optimalSNR(self):
                self.get_optimalSNR = True
                self.log_likelihood()
                self.get_optimalSNR = False
                return self.optimalSNR
            def noise_log_likelihood(self):
                return self.lognl
            def log_likelihood(self):
                if self.get_optimalSNR:
                    self.optimalSNR.clear()
                self.parameters.update({'mass_2': mass1_from_mchirp_q(\
                    self.parameters['chirp_mass'], self.parameters['mass_ratio'])})
                self.parameters.update({'mass_1': mass2_from_mchirp_q(\
                    self.parameters['chirp_mass'], self.parameters['mass_ratio'])})
                parameters = transform_bilby_to_pycbc(self.parameters, \
                    f_ref=self.static_params['f_ref'], return_iota=True)
                if self.useGatedModel:
                    parameters.update(self.gate_parames)
                self.model.update(**parameters)
                logll = self.model._loglikelihood()
                if self.get_optimalSNR:
                    if useGatedModel:
                        for det, h in self.model.get_gated_waveforms().items():
                            slc = slice(self.model._kmin[det], self.model._kmax[det])
                            h[slc] *= np.sqrt(4*self.model._invpsds[det].delta_f*self.model._invpsds[det][slc])
                            hh = h[slc].inner(h[slc]).real 
                            self.optimalSNR.update({det: hh**0.5})
                    else:
                        self.optimalSNR.update({det: (self.model.det_optimal_snrsq(det))**0.5 for det in self.det_names})
                return logll


    if 'inje' in label:
        # assign injection parameters to 1g/2g mergers
        bilby_vari_name = ['theta_jn', 'phi_jl', 'tilt_1', 'tilt_2', 'phi_12', \
                           'a_1', 'a_2', 'mass_1', 'mass_2', 'geocent_time', \
                           'phase', 'psi', 'luminosity_distance', 'ra', 'dec']
        share_par_names = bilby_vari_name[-3:]
        share_par_value = [1000, 1.375, -1.2108]
        if 'inje5' in label:
            share_par_value = [500, 1.375, -1.2108]
        share_parameter = dict(zip(share_par_names, share_par_value))

        first_gen_names = ['1g_'+name for name in bilby_vari_name[:-3]]
        if 'inje1' in label:
            first_gen_value = [0.4, 0.3, 0.5, 1.0, 1.7, 0.1, 0.05, 30, 20, 1126259642.413, 1.3, 2.659]
        elif 'inje2' in label:
            first_gen_value = [0.4, 0.3, 0.5, 1.0, 1.7, 0.1, 0.05, 15, 15, 1126259642.413, 1.3, 2.659]
        elif 'inje3' in label:
            first_gen_value = [0.4, 0.3, 0.5, 1.0, 1.7, 0.1, 0.05, 40, 10, 1126259642.413, 1.3, 2.659]
        elif 'inje4' in label:
            first_gen_value = [0.4, 0.3, 0.5, 1.0, 1.7, 0.3, 0.30, 30, 20, 1126259642.413, 1.3, 2.659]
        elif 'inje5' in label:
            first_gen_value = [1.3, 0.3, 0.5, 1.0, 1.7, 0.1, 0.05, 30, 20, 1126259642.413, 1.3, 2.659]
        else:
            exit()
        first_gen_param = dict(zip(first_gen_names, first_gen_value))

        from pycbc.pnutils import hybrid_meco_frequency
        from bilby.gw.conversion import bilby_to_lalsimulation_spins
        from pycbc.conversions import final_mass_from_initial, final_spin_from_initial
        from pycbc.conversions import mchirp_from_mass1_mass2, q_from_mass1_mass2
        iota_spin_xyz = bilby_to_lalsimulation_spins(*(first_gen_value[:9]+[f_ref, first_gen_value[10]]))
        mass12_spin_xyz = np.array(first_gen_value[7:9])
        mass12_spin_xyz = np.append(mass12_spin_xyz, np.array(iota_spin_xyz[1:]))
        f_hMECO_1g = hybrid_meco_frequency(first_gen_value[7], first_gen_value[8], iota_spin_xyz[3], iota_spin_xyz[6])
        mf = final_mass_from_initial(*mass12_spin_xyz)#, approximant='NRSur7dq4', f_ref=f_ref)
        af = final_spin_from_initial(*mass12_spin_xyz)#, approximant='NRSur7dq4', f_ref=f_ref)
        secon_gen_names = ['2g_'+name for name in bilby_vari_name[:-3]]
        if 'inje1' in label:
            secon_gen_value = [0.5, 2, 0.1, 0.8, 2.0, af, 0.1, mf, 35, first_gen_value[-3]+2*365*24*3600, 1.8, 1.659]
        elif 'inje2' in label:
            secon_gen_value = [0.5, 2, 0.1, 0.8, 2.0, af, 0.1, mf, 20, first_gen_value[-3]+2*365*24*3600, 1.8, 1.659]
        elif 'inje3' in label:
            secon_gen_value = [0.5, 2, 0.1, 0.8, 2.0, af, 0.1, mf, 10, first_gen_value[-3]+2*365*24*3600, 1.8, 1.659]
        elif 'inje4' in label:
            secon_gen_value = [0.5, 2, 0.1, 0.8, 2.0, af, 0.3, mf, 35, first_gen_value[-3]+2*365*24*3600, 1.8, 1.659]
        elif 'inje5' in label:
            secon_gen_value = [1.3, 2, 0.1, 0.8, 2.0, af, 0.1, mf, 35, first_gen_value[-3]+2*365*24*3600, 1.8, 1.659]
        else:
            exit()
        secon_gen_param = dict(zip(secon_gen_names, secon_gen_value))
        iota_spin_xyz = bilby_to_lalsimulation_spins(*(secon_gen_value[:9]+[f_ref, secon_gen_value[10]]))
        f_hMECO_2g = hybrid_meco_frequency(secon_gen_value[7], secon_gen_value[8], iota_spin_xyz[3], iota_spin_xyz[6])

        injection_parameters = {}
        if event_idx[0]:
            fcut_list.append(f_hMECO_1g)
            injection_parameters.update(first_gen_param)
        if event_idx[1]:
            fcut_list.append(f_hMECO_2g)
            injection_parameters.update(secon_gen_param)
        injection_parameters.update(share_parameter)


        # load psd files
        PSDs = {}
        from pycbc.psd import from_txt
        if 'O5' in label:
            asd_filelist = ['AplusDesign', 'AplusDesign', 'avirgo_O5high_NEW', 'kagra_128Mpc', 'AplusDesign']
        else:
            asd_filelist = ['aligo_O4high', 'aligo_O4high', 'avirgo_O4high_NEW', 'kagra_10Mpc']
        for det,fname in zip(det_names, asd_filelist):
            if use_server:
                fpath = '/data/home/public/share/GWRawDataPSD/T2000012-v1/'+fname+'.txt'
            else:
                fpath = '/Users/tangsp/Work/GW/Codes/GW/Frames/PSDs/T2000012-v2/'+fname+'.txt'
            PSDs.update({det: from_txt(fpath, psd_length, 1./duration, f_lower, is_asd_file=True)})


        # generate mock GW strains
        det_names = det_names[:len(asd_filelist)]
        inje_list = []
        if event_idx[0]:
            inje_list.append(dict(zip(bilby_vari_name, first_gen_value+share_par_value)))
        if event_idx[1]:
            inje_list.append(dict(zip(bilby_vari_name, secon_gen_value+share_par_value)))

        from pycbc.noise.gaussian import frequency_noise_from_psd
        noises_list = []
        if event_idx[0]:
            noises_list.append({det: frequency_noise_from_psd(PSDs[det], seed=seedlist[0][i]) for i,det in enumerate(det_names)})
        if event_idx[1]:
            noises_list.append({det: frequency_noise_from_psd(PSDs[det], seed=seedlist[1][i]) for i,det in enumerate(det_names)})

        strains_list = []
        start_time_list = []
        static_params = {'approximant': template_name, 'f_lower': f_lower, 'f_final': f_upper, 'f_ref': f_ref}
        for i,(noise_fd,parameters) in enumerate(zip(noises_list, inje_list)):
            inject_parameters = transform_bilby_to_pycbc(parameters, f_ref=f_ref, return_iota=True)
            time_of_event = parameters['geocent_time']
            start_time = time_of_event - duration + 1
            start_time_list.append(start_time)
            InjectModel = InspiralLikelihood(det_names, start_time, duration, None, None, None, **static_params)
            signals_fd_dict = InjectModel.FDGen_model.generate(**inject_parameters)
            gate_time_list.append(calculate_thMECO(signals_fd_dict, fcut_list[i], \
                inject_parameters, start_time, f_lower, duration, det='H1'))
            strains_fd_dict = {det: signals_fd_dict[det]+noise_fd[det] for det in det_names}
            strains_list.append(strains_fd_dict)


        # construct prior and likelihood
        from bilby.core.prior import Uniform, Constraint, Sine, Cosine, Prior, PriorDict, Interped, PowerLaw
        from bilby.gw.prior import BBHPriorDict, UniformSourceFrame, UniformInComponentsChirpMass, UniformInComponentsMassRatio
        priors = PriorDict()
        priors.clear()
        event_tag_list = []
        if event_idx[0]:
            injection_parameters.update({'1g_chirp_mass': \
                mchirp_from_mass1_mass2(injection_parameters['1g_mass_1'], injection_parameters['1g_mass_2']), \
                '1g_mass_ratio': injection_parameters['1g_mass_2']/injection_parameters['1g_mass_1']})
            event_tag_list.append('1g_')
        if event_idx[1]:
            injection_parameters.update({'2g_chirp_mass': \
                mchirp_from_mass1_mass2(injection_parameters['2g_mass_1'], injection_parameters['2g_mass_2']), \
                '2g_mass_ratio': injection_parameters['2g_mass_2']/injection_parameters['2g_mass_1']})
            event_tag_list.append('2g_')

        for gen_num in event_tag_list:
            priors[gen_num+'mass_1'] = Constraint(name='mass_1', minimum=5.0, maximum=100, unit='$M_{\\odot}$')
            priors[gen_num+'mass_2'] = Constraint(name='mass_2', minimum=5.0, maximum=100, unit='$M_{\\odot}$')
            priors[gen_num+'chirp_mass'] = UniformInComponentsChirpMass(name='chirp_mass', minimum=injection_parameters[gen_num+'chirp_mass']-10, \
                maximum=injection_parameters[gen_num+'chirp_mass']+10, unit='$M_{\\odot}$', latex_label='$\\mathcal{M}^{\\rm '+gen_num[:-1]+'}$')
            priors[gen_num+'mass_ratio'] = UniformInComponentsMassRatio(name='mass_ratio', minimum=0.125, maximum=1., latex_label='$q^{\\rm '+gen_num[:-1]+'}$')
            priors[gen_num+'a_1'] = Uniform(name='a_1', minimum=0, maximum=0.89, boundary='reflective', latex_label='$a_1^{\\rm '+gen_num[:-1]+'}$')
            priors[gen_num+'a_2'] = Uniform(name='a_2', minimum=0, maximum=0.89, boundary='reflective', latex_label='$a_2^{\\rm '+gen_num[:-1]+'}$')
            priors[gen_num+'tilt_1'] = Sine(name='tilt_1', latex_label='$\\theta_1^{\\rm '+gen_num[:-1]+'}$')
            priors[gen_num+'tilt_2'] = Sine(name='tilt_2', latex_label='$\\theta_2^{\\rm '+gen_num[:-1]+'}$')
            priors[gen_num+'phi_12'] = Uniform(name='phi_12', minimum=0, maximum=2 * np.pi, boundary='periodic', latex_label='$\\Delta \\phi^{\\rm '+gen_num[:-1]+'}$')
            priors[gen_num+'phi_jl'] = Uniform(name='phi_jl', minimum=0, maximum=2 * np.pi, boundary='periodic', latex_label='$\\phi_{\\rm JL}^{\\rm '+gen_num[:-1]+'}$')
            priors[gen_num+'theta_jn'] = Sine(name='theta_jn', latex_label='$\\theta_{\\rm JN}^{\\rm '+gen_num[:-1]+'}$')
            priors[gen_num+'geocent_time'] = Uniform(name='geocent_time', minimum=injection_parameters[gen_num+'geocent_time']-0.1, \
                maximum=injection_parameters[gen_num+'geocent_time']+0.1, latex_label='$t_{c}^{\\rm '+gen_num[:-1]+'}$')
            priors[gen_num+'phase'] = Uniform(name='phase', minimum=0, maximum=2 * np.pi, boundary='periodic', latex_label='$\\phi^{\\rm '+gen_num[:-1]+'}$')
            priors[gen_num+'psi'] = Uniform(name='psi', minimum=0, maximum=np.pi, latex_label='$\\psi^{\\rm '+gen_num[:-1]+'}$')
        if useGatedModel:
            priors['ra'], priors['dec'] = injection_parameters['ra'], injection_parameters['dec']
            static_params.update({'ra': injection_parameters['ra'], 'dec': injection_parameters['dec']})
        else:
            priors['ra'] = Uniform(name='ra', minimum=0, maximum=2*np.pi, latex_label='$\\alpha$')
            priors['dec'] = Cosine(name='dec', latex_label='$\\delta$')
        priors['luminosity_distance'] = UniformSourceFrame(name='luminosity_distance', minimum=1, maximum=2500, unit='Mpc', latex_label='$d_{L}$')

        bilby_vari_name = bilby_vari_name[0:7]+bilby_vari_name[9:]
        bilby_vari_name += ['chirp_mass', 'mass_ratio']
        llh_func_list = [InspiralLikelihood(det_names, start_time_list[i], duration, strains_list[i], \
            PSDs, gate_time_list[i], useGatedModel=useGatedModel, **static_params) for i in range(len(event_tag_list))]

        likelihood = MultiEventLikelihood(likelihood_list=llh_func_list, parameters_list=bilby_vari_name, \
            share_par_names=share_par_names, priors=priors, f_ref=f_ref, event_tag_list=event_tag_list, pycbc_params=None)

        likelihood.parameters.update(injection_parameters)
        likelihood.log_likelihood()
        for llk_model in likelihood.likelihood_list:
            print(llk_model.calculate_optimalSNR())

        sampling = 1
        if sampling:
            result = bilby.run_sampler(likelihood=likelihood, priors=priors, use_ratio=use_ratio, live_points=None, sampler='dynesty', \
                npoints=npoints, dlogz=1e-3, sample=sample_method, queue_size=pool_size, injection_parameters=injection_parameters, \
                outdir=outdir, label=label+'_dynesty_'+sample_method)
            result.plot_corner()
