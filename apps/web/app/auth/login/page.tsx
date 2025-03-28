import LoginForm from '@/components/specific/auth/login/login-form';
import SocialLogin from '@/components/specific/auth/social/social-login';
import ActiveMatrix from '@/components/shared/branding/credit/active-matrix';
import FormLogo from '@/components/shared/branding/form-logo/form-logo';

const LoginPage = () => {
	return (
		<div className="screen spacing">
			<FormLogo />
			<LoginForm />
			<SocialLogin />
			<ActiveMatrix />
		</div>
	);
};

export default LoginPage;
